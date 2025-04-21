set(ARM_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ComputeLibrary")

file(MAKE_DIRECTORY "${ARM_DIR}/build")

add_custom_command(
    OUTPUT "${ARM_DIR}/build/libarm_compute.a"
    COMMAND scons Werror=1 debug=0 asserts=0 neon=0 opencl=0 os=linux embed_kernels=1 arch=x86 -j4
    WORKING_DIRECTORY ${ARM_DIR}
)

add_custom_target(arm_compute_lib DEPENDS "${ARM_DIR}/build/libarm_compute.a")

add_executable(Conv ConvolutionLayer.cpp)

include_directories(${ARM_DIR})
include_directories(${ARM_DIR}/include)
target_link_directories(Conv PUBLIC ${ARM_DIR}/build)
target_link_libraries(Conv arm_compute)

add_dependencies(Conv arm_compute_lib)