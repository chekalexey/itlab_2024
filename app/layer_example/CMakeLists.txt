set(ARM_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ComputeLibrary")

include_directories(${ARM_DIR})
link_directories(${ARM_DIR}/build)

add_executable(ConvolutionLayer ConvolutionLayer.cpp)
add_executable(ElementwiseLayer ElementwiseLayer.cpp)

target_link_libraries(ConvolutionLayer arm_compute)
target_link_libraries(ElementwiseLayer arm_compute)