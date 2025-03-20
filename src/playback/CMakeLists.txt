project(playback)

add_executable(${PROJECT_NAME} main.cpp)

# Set up for CUDA compilation
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native)
