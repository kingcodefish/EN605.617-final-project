project(recorder)

add_executable(${PROJECT_NAME} main.cpp)

# target_include_directories(${PROJECT_NAME}
#     PUBLIC
#     ${EXTERNAL_DIR}
#     ${EXTERNAL_DIR}/imgui
# )

target_link_libraries(${PROJECT_NAME}
    PUBLIC
    OpenGL::GL
    glfw
    imgui
)

# Set up for CUDA compilation
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native)
