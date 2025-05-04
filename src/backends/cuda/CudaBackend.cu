#include "CudaBackend.h"

#define NOMINMAX
#include <iostream>
#include <windows.h>

#include <cuda_gl_interop.h>

#include <npp.h>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

// Macro to make CUDA error checking much easier
#define CUDA_CHECK(call)                                                       \
do {                                                                           \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "       \
                  << __FILE__ << ":" << __LINE__ << std::endl;                 \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)


// =============================================================================
// CUDA-accessible Union-Find Structure
// =============================================================================
// Used for maintaining connected components when detecting connected regions of
// an image within CUDA kernels. The __device__ nature of this data structure
// makes it accessible to use concurrently.
// -----------------------------------------------------------------------------
//struct ConcurrentUnionFind
//{
//    thrust::device_vector<int> parent;
//    thrust::device_vector<int> rank;
//    size_t size;
//
//    ConcurrentUnionFind(size_t n) : size(n)
//    {
//        parent.resize(size);
//        rank.resize(size);
//
//        // inits parent to parent[i] = i
//        thrust::sequence(parent.begin(), parent.end());
//        thrust::fill(rank.begin(), rank.end(), 0);
//    }
//
//    ~ConcurrentUnionFind()
//    {
//    }
//
//    // This is non-const find that implements path compression -- be careful not
//    // to have multiple kernels access the same x.
//    __device__ int d_find(int x)
//    {
//        int p = parent[x];
//        //while (p != parent[p])
//        //{
//        //    parent[x] = parent[parent[p]]; // path compression
//        //    x = parent[x];
//        //    p = parent[x];
//        //}
//        return p;
//    }
//
//    __device__ void d_unite(int x, int y)
//    {
//        int rootX = d_find(x);
//        int rootY = d_find(y);
//        if (rootX == rootY) return;
//
//        if (rank[rootX] < rank[rootY])
//        {
//            parent[rootX] = rootY;
//        }
//        else if (rank[rootX] > rank[rootY])
//        {
//            parent[rootY] = rootX;
//        }
//        else
//        {
//            parent[rootY] = rootX;
//            atomicAdd(&thrust::raw_pointer_cast(rank.data())[rootX], 1); // safe increment
//        }
//    }
//
//    __device__ void d_unite_atomic(int x, int y)
//    {
//        while (true)
//        {
//            x = d_find(x);
//            y = d_find(y);
//            if (x == y) break;
//            //if (x < y)
//            //{
//            //    if (atomicCAS(&thrust::raw_pointer_cast(parent.data())[y], y, x) == y) break;
//            //}
//            //else
//            //{
//            //    if (atomicCAS(&thrust::raw_pointer_cast(parent.data())[x], x, y) == x) break;
//            //}
//            break;
//        }
//    }
//};

// Converts the image to grayscale to make it easier to distinguish connected
// components of the image.
__global__ void convertGrayscale(const uint8_t* __restrict__ d_image, uint8_t* __restrict__ d_output, size_t pitch,
    int m_width, int m_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < m_width && y < m_height)
    {
        // RGBA format
        const uint8_t* inPixel = d_image + y * pitch + x * 4;

        // Calculate gray as a weighted average of RGB
        uint8_t gray = 0.299 * inPixel[0] + 0.587 * inPixel[1]
            + 0.114 * inPixel[2];

        // Write results to output buffer
        uint8_t* outPixel = d_output + y * pitch + x * 4;
        outPixel[0] = gray;
        outPixel[1] = gray;
        outPixel[2] = gray;
        outPixel[3] = 255;
    }
}

// Reimplmementation of nppiLabelMarkersUF_8u_C1IR that provides a way of
// preserving the RGBA format of the image for output.
//
// Based on the pseudocode in this paper:
// https://arxiv.org/pdf/1708.08180

__device__ void findAndUnion(int* labelMap, size_t labelPitch, int aX, int aY, int bX, int bY) {
    int* rowA = (int*)((char*)labelMap + aY * labelPitch);
    int* rowB = (int*)((char*)labelMap + bY * labelPitch);

    int rootA = rowA[aX];
    while (true) {
        int* row = (int*)((char*)labelMap + (rootA / labelPitch) * labelPitch);
        int parent = row[rootA % (labelPitch / sizeof(int))];
        if (parent == rootA) break;
        rootA = parent;
    }

    int rootB = rowB[bX];
    while (true) {
        int* row = (int*)((char*)labelMap + (rootB / labelPitch) * labelPitch);
        int parent = row[rootB % (labelPitch / sizeof(int))];
        if (parent == rootB) break;
        rootB = parent;
    }

    if (rootA != rootB) {
        int* row = (int*)((char*)labelMap + (bY * labelPitch));
        row[bX] = rootA;
    }
}

__global__ void localUFMergeWithCoarseLabelingRGBA_pitched(
    const uint8_t* __restrict__ I,
    size_t pitchBytes,
    int imgWidth,
    int imgHeight,
    int* __restrict__ LabelMap,
    size_t labelPitchBytes)
{
    extern __shared__ int shared_mem[];
    int* label_sm = shared_mem;
    uint32_t* dBuff_fsm = (uint32_t*)(shared_mem + blockDim.x * blockDim.y);

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;

    if (gx < imgWidth && gy < imgHeight)
    {
        // Access pitched input row
        const uchar4* row = (const uchar4*)((const uint8_t*)I + gy * pitchBytes);
        int* lblRow = (int*)((char*)LabelMap + gy * labelPitchBytes);

        uchar4 pix = row[gx];
        uint32_t val = (uint32_t)pix.x |
            ((uint32_t)pix.y << 8) |
            ((uint32_t)pix.z << 16) |
            ((uint32_t)pix.w << 24);
        dBuff_fsm[tid] = val;
        label_sm[tid] = tid;
        __syncthreads();

        if (tx > 0 && dBuff_fsm[tid] == dBuff_fsm[tid - 1])
            label_sm[tid] = label_sm[tid - 1];
        __syncthreads();

        if (ty > 0 && dBuff_fsm[tid] == dBuff_fsm[tid - blockDim.x])
            label_sm[tid] = label_sm[tid - blockDim.x];
        __syncthreads();

        int temp = tid;
        while (temp != label_sm[temp]) temp = label_sm[temp];
        label_sm[tid] = temp;
        __syncthreads();

        if (tx > 0 && dBuff_fsm[tid] == dBuff_fsm[tid - 1]) {
            int a = label_sm[tid], b = label_sm[tid - 1];
            if (a < b) label_sm[b] = a; else label_sm[a] = b;
        }
        __syncthreads();
        if (ty > 0 && dBuff_fsm[tid] == dBuff_fsm[tid - blockDim.x]) {
            int a = label_sm[tid], b = label_sm[tid - blockDim.x];
            if (a < b) label_sm[b] = a; else label_sm[a] = b;
        }
        __syncthreads();

        int l = label_sm[tid];
        int root = l;
        while (root != label_sm[root]) root = label_sm[root];
        int lx = l % blockDim.x;
        int ly = l / blockDim.x;
        int global_label = (blockIdx.x * blockDim.x + lx)
            + (blockIdx.y * blockDim.y + ly) * imgWidth;
        lblRow[gx] = global_label;
    }
}

__device__ bool pixelsEqualRGBA(const uint8_t* a, const uint8_t* b) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}

__global__ void connectBoundaries(
    const uint8_t* image, size_t imagePitch,
    int* labelMap, size_t labelPitch,
    int imgWidth, int imgHeight
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= imgWidth || y >= imgHeight) return;

    // Pointers to current pixel
    const uint8_t* currentPixel = image + y * imagePitch + x * 4;

    // Check left neighbor
    if (x > 0 && y < 300) {
        const uint8_t* leftPixel = image + y * imagePitch + (x - 1) * 4;
        if (pixelsEqualRGBA(currentPixel, leftPixel)) {
            findAndUnion(labelMap, labelPitch, x, y, x - 1, y);
        }
    }

    // Check top neighbor
    if (y > 0) {
        const uint8_t* topPixel = image + (y - 1) * imagePitch + x * 4;
        if (pixelsEqualRGBA(currentPixel, topPixel)) {
            findAndUnion(labelMap, labelPitch, x, y, x, y - 1);
        }
    }
}


__device__ uchar4 labelToColor(int label) {
    // arbitrary primes for mixing
    unsigned int x = (unsigned int)label;
    uint8_t r = (uint8_t)((x * 37u) & 0xFF);
    uint8_t g = (uint8_t)((x * 57u >> 8) & 0xFF);
    uint8_t b = (uint8_t)((x * 97u >> 16) & 0xFF);
    return make_uchar4(r, g, b, 255u);
}

__global__ void visualizeLabelMapRGBA_pitched(
    const int* __restrict__ LabelMap,
    size_t labelPitchBytes,
    int width,
    int height,
    uint8_t* __restrict__ OutImage,
    size_t outPitchBytes
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        const int* lblRow = (const int*)((const char*)LabelMap + y * labelPitchBytes);
        uchar4* outRow = (uchar4*)((char*)OutImage + y * outPitchBytes);
        int lbl = lblRow[x];
        outRow[x] = labelToColor(lbl);
    }
}
//__global__ void visualizeLabelMapRGBA(
//    const int* __restrict__ LabelMap,
//    size_t pitch,
//    int width,
//    int height,
//    uint8_t* __restrict__ OutImage
//)
//{
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//    if (x < width && y < height) {
//        int labelIdx = y * width + x;
//        int lbl = LabelMap[labelIdx];
//        uchar4 color = labelToColor(lbl);
//
//        // have to account for pitch of image (not accounted in label calc)
//        int outIdx = y * pitch + x * 4;
//        OutImage[outIdx] = color.x;
//        OutImage[outIdx + 1] = color.y;
//        OutImage[outIdx + 2] = color.z;
//        OutImage[outIdx + 3] = color.w;
//    }
//}

void CudaImageProcessor::processImage()
{
    // An invalid texID means we weren't initialized properly
    if (m_texID == 0)
    {
        return;
    }

    cudaGraphicsResource* cudaResource = nullptr;

    // We need to get CUDA to register the texture we've reserved in OpenGL
    // so that we can conduct image processing on the texture.
    CUDA_CHECK(cudaGraphicsGLRegisterImage(&cudaResource,
        m_texID,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsSurfaceLoadStore));

    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaResource, 0));

    cudaArray_t cudaArray;
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cudaArray,
        cudaResource, 0, 0));

    // Allocate device memory for NPP processing
    uint8_t* d_inBuffer;
    size_t pitch;
    CUDA_CHECK(cudaMallocPitch(&d_inBuffer, &pitch,
        m_width * sizeof(uint8_t) * 4, m_height));

    uint8_t* d_outBuffer;
    size_t pitch_dst;
    CUDA_CHECK(cudaMallocPitch(&d_outBuffer, &pitch_dst,
        m_width * sizeof(uint8_t) * 4, m_height));

    // Copy from surface to device memory (intermediate buffer)
    CUDA_CHECK(cudaMemcpy2DFromArray(d_inBuffer, pitch, cudaArray, 0, 0,
        m_width * sizeof(uint8_t) * 4, m_height, cudaMemcpyDeviceToDevice));

    // Convert image to grayscale
    dim3 blockSize(16, 16);
    dim3 gridSize((m_width + 15) / 16, (m_height + 15) / 16);
    convertGrayscale<<<gridSize, blockSize>>>(d_inBuffer, d_outBuffer,
        pitch_dst, m_width, m_height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Find connected components
    //ConcurrentUnionFind uf(m_width * m_height);
    size_t mapSize = m_width * m_height * sizeof(int);
    int* d_LabelMap;
    size_t labelPitch;
    CUDA_CHECK(cudaMallocPitch(&d_LabelMap, &labelPitch,
        m_width * sizeof(int), m_height));
    dim3 blockDim(16, 16);
    dim3 gridDim((m_width + 15) / 16, (m_height + 15) / 16);
    size_t sharedBytes = (16 * 16) * (sizeof(int) + sizeof(uint32_t));
    localUFMergeWithCoarseLabelingRGBA_pitched <<<gridDim, blockDim, sharedBytes>>>(d_outBuffer, pitch_dst, m_width, m_height, d_LabelMap, labelPitch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    connectBoundaries<<<gridDim, blockDim>>>(d_outBuffer, pitch_dst, d_LabelMap, labelPitch, m_width, m_height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    visualizeLabelMapRGBA_pitched<<<gridDim, blockDim>>>(
        d_LabelMap, labelPitch, m_width, m_height, d_outBuffer, pitch_dst
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result back to the CUDA surface
    CUDA_CHECK(cudaMemcpy2DToArray(cudaArray, 0, 0, d_outBuffer, pitch_dst,
        m_width * sizeof(uint8_t) * 4, m_height, cudaMemcpyDeviceToDevice));

    //GLuint textureID;
    //glGenTextures(1, &textureID);
    //glBindTexture(GL_TEXTURE_2D, 500);

    //// Set texture parameters (adjust as needed)
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //// Upload texture to GPU
    //glTexImage2D(
    //    GL_TEXTURE_2D,
    //    0,
    //    GL_RGBA,
    //    m_width,
    //    m_height,
    //    0,
    //    GL_BGRA_EXT,  // BGRA matches the DIB section format
    //    GL_UNSIGNED_BYTE,
    //    cudaArray
    //);

    // Free the temporary buffer
    CUDA_CHECK(cudaFree(d_inBuffer));
    CUDA_CHECK(cudaFree(d_outBuffer));

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaResource, 0));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaResource));
}