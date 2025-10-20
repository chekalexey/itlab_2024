#ifndef ACL_RESHAPE_LAYER_H
#define ACL_RESHAPE_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

class ReshapeLayer : public Layer {
private:
    NEReshapeLayer reshape;
    bool configured_ = false;

public:
    ReshapeLayer(int id) : Layer(id) {  }

    void configure(const TensorShape& input_shape, TensorShape& output_shape, Tensor& input, Tensor& output) {
        try {
            input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
            output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

            if (!NEReshapeLayer::validate(input.info(), output.info())) {
                throw std::runtime_error("ReshapeLayer: Validation failed");
            }

            input.allocator()->allocate();
            output.allocator()->allocate();

            reshape.configure(&input, &output);
            configured_ = true;
        }
        catch (const std::exception& e) {
            configured_ = false;
            std::cerr << "ReshapeLayer configuration error: " << e.what() << std::endl;
        }
    }

    void exec() override {
        if (!configured_) {
            throw std::runtime_error("ReshapeLayer: Layer not configured.");
        }
        reshape.run();
    }

    std::string get_type_name() const override { return "ReshapeLayer"; }
};

#endif