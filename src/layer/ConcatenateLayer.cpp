#ifndef ACL_CONCATENATE_LAYER_H
#define ACL_CONCATENATE_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

class ConcatenateLayer : public Layer {
private:
    NEConcatenateLayer concat;
    bool configured_ = false;

public:
    ConcatenateLayer(int id): Layer(id) { }

    void configure(const std::vector<TensorShape>& inputs_shapes, unsigned int axis, TensorShape& output_shape, 
        std::vector<Tensor*>& input, Tensor& output) {

        if (inputs_shapes.empty()) {
            throw std::runtime_error("Concat: Input shapes list cannot be empty.");
        }
        if (inputs_shapes.size() != input.size()) {
            throw std::runtime_error("Concat: vector size mismatch.");
        }
        std::vector<const ITensor*> inpcopy;
        for (int i = 0; i < input.size(); i++) {
            input[i]->allocator()->init(TensorInfo(inputs_shapes[i], 1, DataType::F32));
            input[i]->allocator()->allocate();
            inpcopy.push_back(input[i]);
        }
        output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));
        concat.configure(inpcopy, &output, axis);
        output.allocator()->allocate();
        configured_ = true;
    }

    void exec() override {
        if (!configured_) {
            throw std::runtime_error("ConcatenateLayer: Layer not configured.");
        }
        concat.run();
    }

    std::string get_type_name() const override {
        return "ConcatenateLayer";
    }
};

#endif