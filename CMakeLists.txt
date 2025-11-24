#cmake_minimum_required(VERSION 3.27)
#project(Final_Project)
#
#set(CMAKE_CXX_STANDARD 17)
#
#add_executable(Final_Project main.cpp
#        Configs.cpp)
#
## Find and link required frameworks
#find_library(OPENGL_LIBRARY OpenGL)
#find_library(GLUT_LIBRARY GLUT)
#
## Link libraries and packages
#target_link_libraries(Final_Project
#        PUBLIC
#        ${OPENGL_LIBRARY}
#        ${GLUT_LIBRARY}
#        PRIVATE
#        "-framework OpenCL"
#)
cmake_minimum_required(VERSION 3.27.8)
project(Final_Project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create executable first
add_executable(Final_Project main.cpp
        Configs.cpp)

# Include GLEW headers
include_directories(/opt/homebrew/include)

# Link directories (Homebrew)
link_directories(/opt/homebrew/lib)


find_package(OpenCL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)


target_link_libraries(Final_Project PRIVATE
        OpenCL::OpenCL
        OpenGL::GL
        GLUT::GLUT
        /opt/homebrew/lib/libGLEW.dylib
)


