#ifndef ACL_ELEMENTWISE_LAYER_H
#define ACL_ELEMENTWISE_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

using namespace arm_compute;
using namespace utils;

enum class ElementwiseOp { ADD, DIV, ABS, SIGM, SWISH, SQUARED_DIFF };

class ElementwiseLayer : public Layer {
private:
    ElementwiseOp op_type;
    NEActivationLayer act;
    NEArithmeticAddition add;
    NEElementwiseDivision div;
    NEElementwiseSquaredDiff sqdiff;
    bool configured_ = false;

public:
    ElementwiseLayer(int id, ElementwiseOp op) : op_type(op), Layer(id) { }

    ElementwiseLayer() : ElementwiseLayer(0, ElementwiseOp::ADD) { }

    void configure(const TensorShape& input_shape, TensorShape& output_shape, Tensor& input, Tensor& output) {
        try {
            input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
            output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));
            
            input.allocator()->allocate();
            output.allocator()->allocate();
            switch (op_type) {
            case ElementwiseOp::ABS: {
                if (!NEActivationLayer::validate(input.info(), output.info(), 
                    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::ABS))) {
                    throw std::runtime_error("AbsLayer: Validation failed");
                }


                act.configure(&input, &output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::ABS));
                break;
            }
            case ElementwiseOp::SIGM: {
                if (!NEActivationLayer::validate(input.info(), output.info(), 
                    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC))) {
                    throw std::runtime_error("SigmoidLayer: Validation failed");
                }

                act.configure(&input, &output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
                break;
            }
            case ElementwiseOp::SWISH: {
                if (!NEActivationLayer::validate(input.info(), output.info(), 
                    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::SWISH))) {
                    throw std::runtime_error("SwishLayer: Validation failed");
                }

                act.configure(&input, &output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::SWISH));
                break;
            }
            default:
                throw std::runtime_error("ElementwiseLayer: This operation requires two inputs");
            }
            configured_ = true;
        }
        catch (const std::exception& e) {
            configured_ = false;
            std::cerr << "ElementwiseLayer configuration error: " << e.what() << std::endl;
        }
    }

    void configure(const TensorShape& input1_shape, const TensorShape& input2_shape, TensorShape& output_shape, 
        Tensor& input1, Tensor& input2, Tensor& output) {
        try {
            input1.allocator()->init(TensorInfo(input1_shape, 1, DataType::F32));
            input2.allocator()->init(TensorInfo(input2_shape, 1, DataType::F32));
            output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

            input1.allocator()->allocate();
            input2.allocator()->allocate();
            output.allocator()->allocate();
            switch (op_type) {
            case ElementwiseOp::ADD: {
                if (!NEArithmeticAddition::validate(input1.info(), input2.info(), output.info(), ConvertPolicy::WRAP)) {
                    throw std::runtime_error("AddLayer: Validation failed");
                }
                

                add.configure(&input1, &input2, &output, ConvertPolicy::WRAP);
                break;
            }
            case ElementwiseOp::DIV: {
                if (!NEElementwiseDivision::validate(input1.info(), input2.info(), output.info())) {
                    throw std::runtime_error("DivLayer: Validation failed");
                }

                div.configure(&input1, &input2, &output);
                break;
            }
            case ElementwiseOp::SQUARED_DIFF: {
                if (!NEElementwiseSquaredDiff::validate(input1.info(), input2.info(), output.info())) {
                    throw std::runtime_error("SquaredDiffLayer: Validation failed");
                }

                sqdiff.configure(&input1, &input2, &output);
                break;
            }
            default:
                throw std::runtime_error("ElementwiseLayer: This operation requires single input");
            }
            configured_ = true;
        }
        catch (const std::exception& e) {
            configured_ = false;
            std::cerr << "ElementwiseLayer configuration error: " << e.what() << std::endl;
        }
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