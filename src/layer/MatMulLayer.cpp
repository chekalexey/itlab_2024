#ifndef ACL_MATMUL_LAYER_H
#define ACL_MATMUL_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

using namespace arm_compute;
using namespace utils;

class MatMulLayer : public Layer {
private:
    NEMatMul m;
    bool configured_ = false;

public:
    MatMulLayer(int id){
        setID(id);
    }

    void configure(TensorShape& input_x_shape, TensorShape& input_y_shape, TensorShape& output_shape,
        Tensor& input_x, Tensor& input_y, Tensor& output) {

        input_x.allocator()->init(TensorInfo(input_x_shape, 1, DataType::F32));
        input_y.allocator()->init(TensorInfo(input_y_shape, 1, DataType::F32));
        output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

        input_x.allocator()->allocate();
        input_y.allocator()->allocate();
        output.allocator()->allocate();
        m.configure(&input_x, &input_y, &output, MatMulInfo(), CpuMatMulSettings(), ActivationLayerInfo());
        configured_ = true;
    }

    void exec() override {
        if (!configured_) {
            throw std::runtime_error("MatMulLayer: Layer not configured before exec.");
        }
        m.run();
    }

    std::string get_type_name() const override { return "MatMulLayer"; }
};

#endif