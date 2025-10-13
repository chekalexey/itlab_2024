#ifndef ACL_RESIZE_LAYER_H
#define ACL_RESIZE_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

using namespace arm_compute;
using namespace utils;

class ResizeLayer : public Layer {
private:
    NEScale resize;
    bool configured_ = false;

public:
    ResizeLayer(int id) : Layer(id) {  }
    void configure(TensorShape& input_shape, TensorShape& output_shape, Tensor& input, Tensor& output) {
        input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
        output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

        input.allocator()->allocate();
        output.allocator()->allocate();

        resize.configure(&input, &output,
                     ScaleKernelInfo{
                         InterpolationPolicy::NEAREST_NEIGHBOR,
                         BorderMode::REPLICATE,
                         PixelValue(),
                         SamplingPolicy::CENTER,
                     });
        configured_ = true;
    }

    void exec() override {
        if (!configured_) {
            throw std::runtime_error("ResizeLayer: Layer not configured before exec.");
        }
        resize.run();
    }

    std::string get_type_name() const override { return "ResizeLayer"; }
};

#endif