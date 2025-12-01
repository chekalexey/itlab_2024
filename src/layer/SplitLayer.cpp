#ifndef ACL_SPLIT_LAYER_H
#define ACL_SPLIT_LAYER_H

#include <exception>
#include <stdexcept>
#include <string>

#include "layer/layer.h"

class SplitLayer : public Layer {
 private:
  NESplit split_;
  bool configured_ = false;

 public:
  SplitLayer(int id) : Layer(id) {}

  void configure(const TensorShape& input_shape, unsigned int axis,
                 Tensor& input, std::vector<ITensor*>& outputs) {
    try {
      input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
      input.allocator()->allocate();

      std::vector<ITensorInfo*> outputs_info;
      for (auto& output : outputs) {
        if (output != nullptr) {
          outputs_info.push_back(output->info());
        }
      }

      if (!NESplit::validate(input.info(), outputs_info, axis)) {
        throw std::runtime_error("SplitLayer: Validation failed");
      }

      split_.configure(&input, outputs, axis);
      configured_ = true;
    } catch (const std::exception& e) {
      configured_ = false;
      std::cerr << "SplitLayer configuration error: " << e.what() << std::endl;
    }
  }

  void exec() override {
    if (!configured_) {
      throw std::runtime_error("SplitLayer: Layer not configured.");
    }
    split_.run();
  }

  std::string get_type_name() const override { return "SplitLayer"; }
};

#endif