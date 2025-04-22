set(ARM_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ComputeLibrary")

add_executable(Conv ConvolutionLayer.cpp)

include_directories(${ARM_DIR})
include_directories(${ARM_DIR}/include)
target_link_directories(MatMul PUBLIC ${ARM_DIR}/build)

target_link_libraries(Conv arm_compute)

add_dependencies(Conv build_compute_library)