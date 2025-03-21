project(imgui)

# Only included the ones we need
add_library(${PROJECT_NAME} STATIC
  imgui/imgui.h
  imgui/imgui.cpp
  imgui/imgui_internal.h
  imgui/imgui_draw.cpp
  imgui/imgui_tables.cpp
  imgui/imgui_widgets.cpp
  imgui/backends/imgui_impl_glfw.h
  imgui/backends/imgui_impl_glfw.cpp
  imgui/backends/imgui_impl_opengl3.h
  imgui/backends/imgui_impl_opengl3.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/imgui
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
  glfw
)
