#ifndef ACL_CONCATENATE_LAYER_H
#define ACL_CONCATENATE_LAYER_H

#include <exception>
#include <stdexcept>
#include <string>

#include "layer/layer.h"

class ConcatenateLayer : public Layer {
 private:
  NEConcatenateLayer concat_;
  bool configured_ = false;

 public:
  ConcatenateLayer(int id) : Layer(id) {}

  void configure(const std::vector<TensorShape>& inputs_shapes,
                 unsigned int axis, TensorShape& output_shape,
                 std::vector<Tensor*>& input, Tensor& output) {
    try {
      std::vector<const ITensor*> inpcopy;
      std::vector<const ITensorInfo*> inp_info;

      for (int i = 0; i < input.size(); i++) {
        input[i]->allocator()->init(
            TensorInfo(inputs_shapes[i], 1, DataType::F32));
        input[i]->allocator()->allocate();
        inp_info.push_back(input[i]->info());
        inpcopy.push_back(input[i]);
      }

      output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

      if (!NEConcatenateLayer::validate(inp_info, output.info(), axis)) {
        throw std::runtime_error("ConcatenateLayer: Validation failed");
      }

      output.allocator()->allocate();
      concat_.configure(inpcopy, &output, axis);

      configured_ = true;
    } catch (const std::exception& e) {
      configured_ = false;
      std::cerr << "ConcatenateLayer configuration error: " << e.what()
                << std::endl;
    }
  }

  void exec() override {
    if (!configured_) {
      throw std::runtime_error("ConcatenateLayer: Layer not configured.");
    }
    concat_.run();
  }

  std::string get_type_name() const override { return "ConcatenateLayer"; }
};

#endif