project(recorder)

add_executable(${PROJECT_NAME}
    main.cpp
    ContextObserver.h
    backends/Win32ContextObserver.h
    backends/Win32ContextObserver.cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)

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
