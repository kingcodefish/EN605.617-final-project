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


// Code taken from: https://github.com/FolkeV/CUDA_CCL/tree/master
// at an attempt to patch the connected component labeling algorithm.
namespace NOT_MY_CODE
{
    // ---------- Find the root of a chain ----------
    __device__ __inline__ unsigned int find_root(unsigned int* labels,
        unsigned int label)
    {
        // Resolve Label
        unsigned int next = labels[label];

        // Follow chain
        while (label != next) {
            // Move to next
            label = next;
            next = labels[label];
        }

        // Return label
        return(label);
    }

    // ---------- Label Reduction ----------
    __device__ __inline__ unsigned int reduction(unsigned int* g_labels,
        unsigned int label1, unsigned int label2)
    {
        // Get next labels
        unsigned int next1 = (label1 != label2) ? g_labels[label1] : 0;
        unsigned int next2 = (label1 != label2) ? g_labels[label2] : 0;

        // Find label1
        while ((label1 != label2) && (label1 != next1))
        {
            // Adopt label
            label1 = next1;

            // Fetch next label
            next1 = g_labels[label1];
        }

        // Find label2
        while ((label1 != label2) && (label2 != next2))
        {
            // Adopt label
            label2 = next2;

            // Fetch next label
            next2 = g_labels[label2];
        }

        unsigned int label3;
        // While Labels are different
        while (label1 != label2)
        {
            // Label 2 should be smallest
            if (label1 < label2)
            {
                // Swap Labels
                label1 = label1 ^ label2;
                label2 = label1 ^ label2;
                label1 = label1 ^ label2;
            }

            // AtomicMin label1 to label2
            label3 = atomicMin(&g_labels[label1], label2);
            label1 = (label1 == label3) ? label2 : label3;
        }

        // Return label1
        return(label1);
    }

    __global__ void init_labels(unsigned int* g_labels, const uint8_t* g_image,
        const size_t numCols, const size_t numRows)
    {
        // Calculate index
        const unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
        const unsigned int iy = (blockIdx.y * blockDim.y) + threadIdx.y;

        // Check Thread Range
        if ((ix < numCols) && (iy < numRows))
        {
            // Fetch five image values
            const uint8_t pyx = g_image[iy * numCols + ix];

            // Neighbour Connections
            const bool nym1x = (iy > 0) ? (pyx == g_image[(iy - 1) * numCols + ix]) : false;
            const bool nyxm1 = (ix > 0) ? (pyx == g_image[(iy)*numCols + ix - 1]) : false;
            const bool nym1xm1 = ((iy > 0) && (ix > 0)) ? (pyx == g_image[(iy - 1) * numCols + ix - 1]) : false;
            const bool nym1xp1 = ((iy > 0) && (ix < numCols - 1)) ? (pyx == g_image[(iy - 1) * numCols + ix + 1]) : false;

            // Label
            unsigned int label;

            // Initialise Label
            // Label will be chosen in the following order:
            // NW > N > NE > E > current position
            label = (nyxm1) ? iy * numCols + ix - 1 : iy * numCols + ix;
            label = (nym1xp1) ? (iy - 1) * numCols + ix + 1 : label;
            label = (nym1x) ? (iy - 1) * numCols + ix : label;
            label = (nym1xm1) ? (iy - 1) * numCols + ix - 1 : label;

            // Write to Global Memory
            g_labels[iy * numCols + ix] = label;
        }
    }

    // Resolve Kernel
    __global__ void resolve_labels(unsigned int* g_labels,
        const size_t numCols, const size_t numRows)
    {
        // Calculate index
        const unsigned int id = ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols +
            ((blockIdx.x * blockDim.x) + threadIdx.x);

        // Check Thread Range
        if (id < (numRows * numCols))
        {
            // Resolve Label
            g_labels[id] = find_root(g_labels, g_labels[id]);
        }
    }

    // Label Reduction
    __global__ void label_reduction(unsigned int* g_labels, const uint8_t* g_image,
        const size_t numCols, const size_t numRows)
    {
        // Calculate index
        const unsigned int iy = ((blockIdx.y * blockDim.y) + threadIdx.y);
        const unsigned int ix = ((blockIdx.x * blockDim.x) + threadIdx.x);

        // Check Thread Range
        if ((ix < numCols) && (iy < numRows))
        {
            // Compare Image Values
            const uint8_t pyx = g_image[iy * numCols + ix];
            const bool nym1x = (iy > 0) ? (pyx == g_image[(iy - 1) * numCols + ix]) : false;

            if (!nym1x)
            {
                // Neighbouring values
                const bool nym1xm1 = ((iy > 0) && (ix > 0)) ? (pyx == g_image[(iy - 1) * numCols + ix - 1]) : false;
                const bool nyxm1 = (ix > 0) ? (pyx == g_image[(iy)*numCols + ix - 1]) : false;
                const bool nym1xp1 = ((iy > 0) && (ix < numCols - 1)) ? (pyx == g_image[(iy - 1) * numCols + ix + 1]) : false;

                if (nym1xp1)
                {
                    // Check Criticals
                    // There are three cases that need a reduction
                    if ((nym1xm1 && nyxm1) || (nym1xm1 && !nyxm1))
                    {
                        // Get labels
                        unsigned int label1 = g_labels[(iy)*numCols + ix];
                        unsigned int label2 = g_labels[(iy - 1) * numCols + ix + 1];

                        // Reduction
                        reduction(g_labels, label1, label2);
                    }

                    if (!nym1xm1 && nyxm1)
                    {
                        // Get labels
                        unsigned int label1 = g_labels[(iy)*numCols + ix];
                        unsigned int label2 = g_labels[(iy)*numCols + ix - 1];

                        // Reduction
                        reduction(g_labels, label1, label2);
                    }
                }
            }
        }
    }

    // Force background to get label zero;
    __global__ void resolve_background(unsigned int* g_labels, const uint8_t* g_image,
        const size_t numCols, const size_t numRows)
    {
        // Calculate index
        const unsigned int id = ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols +
            ((blockIdx.x * blockDim.x) + threadIdx.x);

        if (id < numRows * numCols) {
            g_labels[id] = (g_image[id] > 0) ? g_labels[id] + 1 : 0;
        }
    }
}

// This was a bad attempt :(
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
__global__ void convertGrayscale(const uint8_t* __restrict__ d_image,
    uint8_t* __restrict__ d_output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // RGBA format
        const size_t pixel = (y * width + x) * sizeof(uint8_t) * 4;

        // Calculate gray as a weighted average of RGB
        uint8_t gray = 0.299 * d_image[pixel + 0] + 0.587 * d_image[pixel + 1]
            + 0.114 * d_image[pixel + 2];

        // Write results to output buffer
        d_output[pixel + 0] = gray;
        d_output[pixel + 1] = gray;
        d_output[pixel + 2] = gray;
        d_output[pixel + 3] = 255;
    }
}

__device__ uchar4 labelToColor(unsigned int label)
{
    // arbitrary primes for mixing
    uint8_t r = (uint8_t)((label * 37u) & 0xFF);
    uint8_t g = (uint8_t)((label * 57u >> 8) & 0xFF);
    uint8_t b = (uint8_t)((label * 97u >> 16) & 0xFF);
    return make_uchar4(r, g, b, 255u);
}

__global__ void labels_to_image(
    const unsigned int* __restrict__ LabelMap,
    uint8_t* __restrict__ OutImage,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        size_t pixel = y * width + x;
        uchar4 outColor = labelToColor(LabelMap[pixel]);
        ((uchar4*)(OutImage + pixel * sizeof(uint8_t) * 4))[0] = outColor;
    }
}

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
    CUDA_CHECK(cudaMalloc(&d_inBuffer,
        m_width * sizeof(uint8_t) * 4 * m_height));

    uint8_t* d_outBuffer;
    CUDA_CHECK(cudaMalloc(&d_outBuffer,
        m_width * sizeof(uint8_t) * 4 * m_height));

    // Copy from surface to device memory (intermediate buffer)
    CUDA_CHECK(cudaMemcpyFromArray(d_inBuffer, cudaArray, 0, 0,
        m_width * sizeof(uint8_t) * 4 * m_height, cudaMemcpyDeviceToDevice));
 
    size_t mapSize = m_width * m_height * sizeof(unsigned int);
    unsigned int* d_LabelMap;
    CUDA_CHECK(cudaMalloc(&d_LabelMap, mapSize));

    dim3 block(16, 16);
    dim3 grid((m_width + 15) / 16, (m_height + 15) / 16);

    convertGrayscale<<<grid, block>>>(d_inBuffer, d_outBuffer, m_width,
        m_height);
    CUDA_CHECK(cudaGetLastError());

    // Initialize labels
    NOT_MY_CODE::init_labels<<<grid, block>>>(d_LabelMap, d_inBuffer, m_width,
        m_height);
    CUDA_CHECK(cudaGetLastError());

    // Analysis
    NOT_MY_CODE::resolve_labels<<<grid, block>>>(d_LabelMap, m_width, m_height);
    CUDA_CHECK(cudaGetLastError());

    // Label Reduction
    NOT_MY_CODE::label_reduction<<<grid, block>>>(d_LabelMap, d_inBuffer, m_width,
        m_height);
    CUDA_CHECK(cudaGetLastError());

    // Analysis
    NOT_MY_CODE::resolve_labels<<<grid, block>>>(d_LabelMap, m_width, m_height);
    CUDA_CHECK(cudaGetLastError());

    // Force background to have label zero;
    NOT_MY_CODE::resolve_background<<<grid, block>>>(d_LabelMap, d_inBuffer,
        m_width, m_height);
    CUDA_CHECK(cudaGetLastError());

    labels_to_image<<<grid, block>>>(d_LabelMap, d_outBuffer, m_width,
        m_height);
    CUDA_CHECK(cudaGetLastError());

    // Copy the result back to the CUDA surface
    CUDA_CHECK(cudaMemcpyToArray(cudaArray, 0, 0, d_outBuffer,
        m_width * sizeof(uint8_t) * 4 * m_height, cudaMemcpyDeviceToDevice));

    // Free the temporary buffer
    CUDA_CHECK(cudaFree(d_inBuffer));
    CUDA_CHECK(cudaFree(d_outBuffer));

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaResource, 0));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaResource));
}