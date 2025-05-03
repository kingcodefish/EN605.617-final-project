cmake_minimum_required(VERSION 3.24)
project(EN_605_617 CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)

# Build external libs
add_subdirectory(external)

# System library
find_package(OpenGL REQUIRED)

add_subdirectory(src)
