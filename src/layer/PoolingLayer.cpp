#ifndef ACL_POOLING_LAYER_H
#define ACL_POOLING_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

class PoolingLayer : public Layer {
private:
    NEPoolingLayer pool;
    bool configured_ = false;

public:
    PoolingLayer(int id) : Layer(id) { }

    void configure(TensorShape& input_shape, 
        TensorShape& output_shape, Tensor& input, Tensor& output) {
        try {
            input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
            output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

            if (!NEPoolingLayer::validate(input.info(), output.info(), PoolingLayerInfo(PoolingType::MAX, DataLayout::NHWC))) {
                throw std::runtime_error("PoolingLayer: Validation failed");
            }

            input.allocator()->allocate();
            output.allocator()->allocate();

            pool.configure(&input, &output, PoolingLayerInfo(PoolingType::MAX, DataLayout::NHWC));
            
            configured_ = true;
        }
        catch (const std::exception& e) {
            configured_ = false;
            std::cerr << "PoolingLayer configuration error: " << e.what() << std::endl;
        }
        
    }

    void exec() override {
        if (!configured_) {
            throw std::runtime_error("PoolingLayer: Layer not configured before exec.");
        }
        pool.run();
    }

    std::string get_type_name() const override { return "PoolingLayer"; }
};

#endif