project(ui)

add_library(${PROJECT_NAME} SHARED
  Context.h
  Context.cpp
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
  OpenGL::GL
  glfw
  imgui
)