project(backends)

add_library(${PROJECT_NAME} SHARED
    backends.h
    ImageProcessor.h
    ImageProcessor.cpp
    cuda/CudaBackend.h
    cuda/CudaBackend.cu
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
    OpenGL::GL
    nppc
    nppif
)

# Set up for CUDA compilation
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native)

# For DLL exports/imports
target_compile_definitions(${PROJECT_NAME} PRIVATE BACKENDS_EXPORTS)
