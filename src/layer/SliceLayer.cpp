#ifndef ACL_SLICE_LAYER_H
#define ACL_SLICE_LAYER_H

#include <exception>
#include <stdexcept>
#include <string>

#include "layer/layer.h"

class SliceLayer : public Layer {
 private:
  NESlice slice_;
  bool configured_ = false;

 public:
  SliceLayer(int id) : Layer(id) {}

  void configure(const TensorShape& input_shape, Coordinates starts,
                 Coordinates ends, TensorShape& output_shape, Tensor& input,
                 Tensor& output) {
    try {
      input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
      output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

      if (!NESlice::validate(input.info(), output.info(), starts, ends)) {
        throw std::runtime_error("SliceLayer: Validation failed");
      }

      input.allocator()->allocate();
      output.allocator()->allocate();

      slice_.configure(&input, &output, starts, ends);
      configured_ = true;
    } catch (const std::exception& e) {
      configured_ = false;
      std::cerr << "SliceLayer configuration error: " << e.what() << std::endl;
    }
  }

  void exec() override {
    if (!configured_) {
      throw std::runtime_error("SliceLayer: Layer not configured.");
    }
    slice_.run();
  }
  std::string get_type_name() const override { return "SliceLayer"; }
};

#endif