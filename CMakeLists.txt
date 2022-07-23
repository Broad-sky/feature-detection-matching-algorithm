cmake_minimum_required(VERSION 2.6)
project(IR)
add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

include_directories(${PROJECT_SOURCE_DIR}/include)
# link_directories(${PROJECT_SOURCE_DIR}/lib)


# cuda
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)

# tensorrt
include_directories(/usr/local/TensorRT-7.2.2.3/include)
link_directories(/usr/local/TensorRT-7.2.2.3/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SRC
src/local_match.cpp
src/akaze_match.cpp
src/match.cpp
src/example.cpp
        )

add_executable(IR ${SRC})

target_link_libraries(IR nvinfer)
target_link_libraries(IR nvinfer_plugin)
target_link_libraries(IR cudart)
target_link_libraries(IR ${OpenCV_LIBS})
