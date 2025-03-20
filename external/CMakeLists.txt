# GLFW config
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build the GLFW example programs" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Build the GLFW test programs" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "Generate installation target" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "Disable building GLFW docs" FORCE)

add_subdirectory(glfw)
add_subdirectory(imgui)
