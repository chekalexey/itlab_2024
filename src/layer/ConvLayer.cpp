#ifndef ACL_CONVOLUTION_LAYER_SIMPLIFIED_H
#define ACL_CONVOLUTION_LAYER_SIMPLIFIED_H

#include <numeric> 
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

class ConvolutionLayer : public Layer {
private:
    NEConvolutionLayer conv;
    bool configured_ = false;

public:
    ConvolutionLayer(int id) : Layer(id) { }

    void configure(
        const TensorShape& input_shape,    
        const TensorShape& weights_shape,
        const TensorShape& biases_shape,
        TensorShape& output_shape,
        const PadStrideInfo& info,
        Tensor& input,
        Tensor& weights,
        Tensor& biases,
        Tensor& output
    ) {

        input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
        weights.allocator()->init(TensorInfo(weights_shape, 1, DataType::F32));
        biases.allocator()->init(TensorInfo(biases_shape, 1, DataType::F32));
        output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

        input.allocator()->allocate();
        weights.allocator()->allocate();
        biases.allocator()->allocate();
        output.allocator()->allocate();
        
        conv.configure(&input, &weights, &biases, &output, info);
        configured_ = true;
    }

    void exec() override {
        if (!configured_) {
            throw std::runtime_error("ConvolutionLayer: Layer not configured.");
        }
        conv.run();
    }

    std::string get_type_name() const override {
        return "ConvolutionLayer";
    }
};

#endif 