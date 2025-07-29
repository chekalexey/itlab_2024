#ifndef ACL_ELEMENTWISE_LAYER_H
#define ACL_ELEMENTWISE_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

using namespace arm_compute;
using namespace utils;

enum class ElementwiseOp {
    ADD,
    DIV,
    ABS,
    SIGM,
    SWISH,
    SQUARED_DIFF
};

class ElementwiseLayer : public Layer {
private:
    ElementwiseOp op_type;
    NEActivationLayer act;
    NEArithmeticAddition add;
    NEElementwiseDivision div;
    NEElementwiseSquaredDiff sqdiff;
    bool configured_ = false;

public:
    ElementwiseLayer(int id, ElementwiseOp op) : op_type(op) { setID(id); }

    ElementwiseLayer() : ElementwiseLayer(0, ElementwiseOp::ADD) { }

    void configure(const TensorShape& input_shape, TensorShape& output_shape, Tensor& input, Tensor& output) {
        input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
        output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

        input.allocator()->allocate();
        output.allocator()->allocate();

        switch (op_type) {
            case ElementwiseOp::ABS: {
                act.configure(&input, &output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::ABS));
                act.run();
                break;
            }
            case ElementwiseOp::SIGM: {
                act.configure(&input, &output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
                act.run();
                break;
            }
            case ElementwiseOp::SWISH: {
                act.configure(&input, &output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::SWISH));
                act.run();
                break;
            }
            default:
                throw std::runtime_error("ElementwiseLayer: This operation requires two inputs");
        }
        configured_ = true;
    }

    void configure(const TensorShape& input1_shape, const TensorShape& input2_shape, TensorShape& output_shape, 
        Tensor& input1, Tensor& input2, Tensor& output) {
        if (input1_shape.total_size() != input2_shape.total_size()) {
            throw std::runtime_error(
                "ElementwiseLayer: Input shapes must have same total size");
        }
        input1.allocator()->init(TensorInfo(input1_shape, 1, DataType::F32));
        input2.allocator()->init(TensorInfo(input2_shape, 1, DataType::F32));
        output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

        input1.allocator()->allocate();
        input2.allocator()->allocate();
        output.allocator()->allocate();

        switch (op_type) {
            case ElementwiseOp::ADD: {
                add.configure(&input1, &input2, &output, ConvertPolicy::WRAP);
                add.run();
                break;
            }
            case ElementwiseOp::DIV: {
                div.configure(&input1, &input2, &output);
                div.run();
                break;
            }
            case ElementwiseOp::SQUARED_DIFF: {
                sqdiff.configure(&input1, &input2, &output);
                sqdiff.run();
                break;
            }
            default:
                throw std::runtime_error("ElementwiseLayer: This operation requires single input");
        }
        configured_ = true;
    }

    void exec() override {
        if (!configured_) {
            throw std::runtime_error("ElementwiseLayer: Layer not configured before exec.");
        }
        switch (op_type) {
            case ElementwiseOp::ABS:
            case ElementwiseOp::SIGM:
            case ElementwiseOp::SWISH:
                act.run();
                break;
            case ElementwiseOp::ADD: {
                add.run();
                break;
            }
            case ElementwiseOp::DIV: {
                div.run();
                break;
            }
            case ElementwiseOp::SQUARED_DIFF: {
                sqdiff.run();
                break;
            }
            default:
                throw std::runtime_error("ElementwiseLayer: This operation requires single input");
        }
    }

    std::string get_type_name() const override {
        switch (op_type) {
            case ElementwiseOp::ADD: return "ElementwiseAddLayer";
            case ElementwiseOp::DIV: return "ElementwiseDivLayer";
            case ElementwiseOp::ABS: return "ElementwiseAbsLayer";
            case ElementwiseOp::SIGM: return "ElementwiseSigmoidLayer";
            case ElementwiseOp::SWISH: return "ElementwiseSwishLayer";
            case ElementwiseOp::SQUARED_DIFF: return "ElementwiseSquaredDiffLayer";
            default:return "ElementwiseUnknownLayer";
        }
    }
};

#endif