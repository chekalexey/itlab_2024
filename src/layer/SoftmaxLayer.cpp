#ifndef ACL_SOFTMAX_LAYER_H
#define ACL_SOFTMAX_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

using namespace arm_compute;
using namespace utils;

class SoftmaxLayer : public Layer {
private:
    NESoftmaxLayer m;
    bool configured_ = false;

public:
    SoftmaxLayer(int id) : Layer(id) { }

    void configure(TensorShape& input_shape, TensorShape& output_shape, Tensor& input, Tensor& output) {
        try {
            input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
            output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

            if (!NESoftmaxLayer::validate(input.info(), output.info())) {
                throw std::runtime_error("SoftmaxLayer: Validation failed");
            }

            input.allocator()->allocate();
            output.allocator()->allocate();

            m.configure(&input, &output);
            configured_ = true;
        }
        catch (const std::exception& e) {
            configured_ = false;
            std::cerr << "SoftmaxLayer configuration error: " << e.what() << std::endl;
        }
    }

    void exec() override {
        if (!configured_) {
            throw std::runtime_error("SoftmaxLayer: Layer not configured before exec.");
        }
        m.run();
    }

    std::string get_type_name() const override { return "SoftmaxLayer"; }
};

#endif