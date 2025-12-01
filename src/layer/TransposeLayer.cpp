#ifndef ACL_TRANSPOSE_LAYER_H
#define ACL_TRANSPOSE_LAYER_H

#include <exception>
#include <stdexcept>
#include <string>

#include "layer/layer.h"

using namespace arm_compute;
using namespace utils;

class TransposeLayer : public Layer {
 private:
  NETranspose t_;
  bool configured_ = false;

 public:
  TransposeLayer(int id) : Layer(id) {}

  void configure(TensorShape& input_shape, TensorShape& output_shape,
                 Tensor& input, Tensor& output) {
    try {
      input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
      output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

      if (!NETranspose::validate(input.info(), output.info())) {
        throw std::runtime_error("TransposeLayer: Validation failed");
      }

      input.allocator()->allocate();
      output.allocator()->allocate();

      t_.configure(&input, &output);
      configured_ = true;
    } catch (const std::exception& e) {
      configured_ = false;
      std::cerr << "TransposeLayer configuration error: " << e.what()
                << std::endl;
    }
  }

  void exec() override {
    if (!configured_) {
      throw std::runtime_error(
          "TransposeLayer: Layer not configured before exec.");
    }
    t_.run();
  }

  std::string get_type_name() const override { return "TransposeLayer"; }
};

#endif