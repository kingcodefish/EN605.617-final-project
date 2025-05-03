#include "CudaBackend.h"

#define NOMINMAX
#include <iostream>
#include <windows.h>

#include <cuda_gl_interop.h>

#include <npp.h>

// Macro to make CUDA error checking much easier
#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "     \
                  << __FILE__ << ":" << __LINE__ << std::endl;               \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while (0)

__global__ void customSobelFilter(uint8_t * d_image, uint8_t * d_output, size_t pitch, NppiSize roiSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < roiSize.width - 1 && y < roiSize.height - 1 && x > 0 && y > 0) {
        uint8_t* pixel = d_image + y * pitch + x * 4;

        // Sobel Kernels for X and Y gradients
        const int sobelX[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
        const int sobelY[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

        int gradientX = 0, gradientY = 0;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                uint8_t* neighbor = d_image + (y + ky) * pitch + (x + kx) * 4;
                gradientX += neighbor[0] * sobelX[ky + 1][kx + 1]; // Using Red Channel
                gradientY += neighbor[0] * sobelY[ky + 1][kx + 1]; // Using Red Channel
            }
        }

        // Calculate gradient magnitude and clamp values
        uint8_t r = min(255, abs(gradientX));  // Horizontal edges in red
        uint8_t g = min(255, abs(gradientY));  // Vertical edges in green
        uint8_t b = min(255.f, sqrtf(gradientX * gradientX + gradientY * gradientY)); // Overall edge strength

        // Write results to output buffer
        uint8_t* outPixel = d_output + y * pitch + x * 4;
        outPixel[0] = r;
        outPixel[1] = g;
        outPixel[2] = b;
        outPixel[3] = 255; // Preserve Alpha
    }
}

void CudaImageProcessor::sobelFilter()
{
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
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0));

    // Allocate device memory for NPP processing
    uint8_t* d_nppBuffer;
    size_t pitch;
    CUDA_CHECK(cudaMallocPitch(&d_nppBuffer, &pitch, m_width * sizeof(uint8_t) * 4, m_height));

    uint8_t* d_nppBuffer_dst;
    size_t pitch_dst;
    CUDA_CHECK(cudaMallocPitch(&d_nppBuffer_dst, &pitch_dst, m_width * sizeof(uint8_t) * 4, m_height));

    // Copy from surface to device memory (intermediate buffer)
    CUDA_CHECK(cudaMemcpy2DFromArray(d_nppBuffer, pitch, cudaArray, 0, 0, m_width * sizeof(uint8_t) * 4, m_height, cudaMemcpyDeviceToDevice));

    // Apply NPP function
    NppiSize roiSize = { m_width, m_height };
    //NppStatus status = nppiFilterSobelVert_8u_C4R(d_nppBuffer, pitch, d_nppBuffer_dst, pitch_dst, roiSize);
    //if (status != NPP_SUCCESS)
    //{
    //    fprintf(stderr, "NPP Error: %d at %s:%d\n", status, __FILE__, __LINE__);
    //    exit(EXIT_FAILURE);
    //}
    dim3 blockSize(16, 16);
    dim3 gridSize((roiSize.width + 15) / 16, (roiSize.height + 15) / 16);
    customSobelFilter << <gridSize, blockSize >> > (d_nppBuffer, d_nppBuffer_dst, pitch, roiSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result back to the CUDA surface
    CUDA_CHECK(cudaMemcpy2DToArray(cudaArray, 0, 0, d_nppBuffer_dst, pitch, m_width * sizeof(uint8_t) * 4, m_height, cudaMemcpyDeviceToDevice));

    // Free the temporary buffer
    CUDA_CHECK(cudaFree(d_nppBuffer));
    CUDA_CHECK(cudaFree(d_nppBuffer_dst));

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaResource, 0));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaResource));
}