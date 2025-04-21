set(ARM_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ComputeLibrary")

file(MAKE_DIRECTORY "${ARM_DIR}/build")

add_custom_command(
    OUTPUT "${ARM_DIR}/build/libarm_compute.so"
    COMMAND scons Werror=1 -j8 debug=0 neon=1 opencl=0 os=linux arch=armv8a build=native
    WORKING_DIRECTORY ${ARM_DIR}
)

add_custom_target(arm_compute_lib DEPENDS "${ARM_DIR}/build/libarm_compute.so")

add_executable(Conv ConvolutionLayer.cpp)

include_directories(${ARM_DIR})
include_directories(${ARM_DIR}/include)
target_link_directories(Conv PUBLIC ${ARM_DIR}/build)
target_link_libraries(Conv arm_compute)

add_dependencies(Conv arm_compute_lib)