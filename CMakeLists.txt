cmake_minimum_required(VERSION 3.27)
project(Final_Project)

set(CMAKE_CXX_STANDARD 17)

add_executable(Final_Project main.cpp
        Configs.cpp)

# Find and link required frameworks
find_library(OPENGL_LIBRARY OpenGL)
find_library(GLUT_LIBRARY GLUT)

# Link libraries and packages
target_link_libraries(Final_Project
        PUBLIC
        ${OPENGL_LIBRARY}
        ${GLUT_LIBRARY}
        PRIVATE
        "-framework OpenCL"
)
