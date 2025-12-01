#ifndef ACL_RESIZE_LAYER_H
#define ACL_RESIZE_LAYER_H

#include <exception>
#include <stdexcept>
#include <string>

#include "layer/layer.h"

using namespace arm_compute;
using namespace utils;

class ResizeLayer : public Layer {
 private:
  NEScale resize_;
  bool configured_ = false;

 public:
  ResizeLayer(int id) : Layer(id) {}
  void configure(TensorShape& input_shape, TensorShape& output_shape,
                 Tensor& input, Tensor& output) {
    try {
      input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
      output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

      if (!NEScale::validate(
              input.info(), output.info(),
              ScaleKernelInfo{InterpolationPolicy::NEAREST_NEIGHBOR,
                              BorderMode::REPLICATE, PixelValue(),
                              SamplingPolicy::CENTER})) {
        throw std::runtime_error("ResizeLayer: Validation failed");
      }

      input.allocator()->allocate();
      output.allocator()->allocate();

      resize_.configure(&input, &output,
                        ScaleKernelInfo{InterpolationPolicy::NEAREST_NEIGHBOR,
                                        BorderMode::REPLICATE, PixelValue(),
                                        SamplingPolicy::CENTER});
      configured_ = true;
    } catch (const std::exception& e) {
      configured_ = false;
      std::cerr << "ResizeLayer configuration error: " << e.what() << std::endl;
    }
  }

  void exec() override {
    if (!configured_) {
      throw std::runtime_error(
          "ResizeLayer: Layer not configured before exec.");
    }
    resize_.run();
  }

  std::string get_type_name() const override { return "ResizeLayer"; }
};

#endif