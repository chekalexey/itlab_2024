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
    MatMulLayer(int id) : Layer(id) { };

    void configure(TensorShape& input_x_shape, TensorShape& input_y_shape, TensorShape& output_shape,
        Tensor& input_x, Tensor& input_y, Tensor& output) {

        try {
            input_x.allocator()->init(TensorInfo(input_x_shape, 1, DataType::F32));
            input_y.allocator()->init(TensorInfo(input_y_shape, 1, DataType::F32));
            output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

            if (!NEMatMul::validate(input_x.info(), input_y.info(), output.info(), MatMulInfo(), CpuMatMulSettings(), ActivationLayerInfo())) {
                throw std::runtime_error("MatMulLayer: Validation failed");
            }

            input_x.allocator()->allocate();
            input_y.allocator()->allocate();
            output.allocator()->allocate();

            m.configure(&input_x, &input_y, &output, MatMulInfo(), CpuMatMulSettings(), ActivationLayerInfo());
            configured_ = true;
        }
        catch (const std::exception& e) {
            configured_ = false;
            std::cerr << "MatMulLayer configuration error: " << e.what() << std::endl;
        }
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