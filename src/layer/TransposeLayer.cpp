#ifndef ACL_TRANSPOSE_LAYER_H
#define ACL_TRANSPOSE_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

using namespace arm_compute;
using namespace utils;

class TransposeLayer : public Layer {
private:
    NETranspose t;
    bool configured_ = false;

public:
    TransposeLayer(int id) {
        setID(id);
    }

    void configure(TensorShape& input_shape, TensorShape& output_shape, Tensor& input, Tensor& output) {
        input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
        output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

        input.allocator()->allocate();
        output.allocator()->allocate();

        t.configure(&input, &output);
        configured_ = true;
    }

    void exec() override {
        if (!configured_) {
            throw std::runtime_error("TransposeLayer: Layer not configured before exec.");
        }
        t.run();
    }

    std::string get_type_name() const override { return "TransposeLayer"; }
};

#endif