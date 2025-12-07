#ifndef ACL_ELEMENTWISE_LAYER_H
#define ACL_ELEMENTWISE_LAYER_H

#include <exception>
#include <stdexcept>
#include <string>

#include "layer/layer.h"

using namespace arm_compute;
using namespace utils;

enum class ElementwiseOp : std::uint8_t {
  kAdd,
  kDiv,
  kAbs,
  kSigm,
  kSwish,
  kSquaredDiff
};

class ElementwiseLayer : public Layer {
 private:
  ElementwiseOp op_type_;
  NEActivationLayer act_;
  NEArithmeticAddition add_;
  NEElementwiseDivision div_;
  NEElementwiseSquaredDiff sqdiff_;
  bool configured_ = false;

 public:
  ElementwiseLayer(int id, ElementwiseOp op) : op_type_(op), Layer(id) {}

  ElementwiseLayer() : ElementwiseLayer(0, ElementwiseOp::kAdd) {}

  void configure(const TensorShape& input_shape, TensorShape& output_shape,
                 Tensor& input, Tensor& output) {
    try {
      input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
      output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

      input.allocator()->allocate();
      output.allocator()->allocate();
      switch (op_type_) {
        case ElementwiseOp::kAbs: {
          if (!NEActivationLayer::validate(
                  input.info(), output.info(),
                  ActivationLayerInfo(
                      ActivationLayerInfo::ActivationFunction::ABS))) {
            throw std::runtime_error("AbsLayer: Validation failed");
          }

          act_.configure(&input, &output,
                         ActivationLayerInfo(
                             ActivationLayerInfo::ActivationFunction::ABS));
          break;
        }
        case ElementwiseOp::kSigm: {
          if (!NEActivationLayer::validate(
                  input.info(), output.info(),
                  ActivationLayerInfo(
                      ActivationLayerInfo::ActivationFunction::LOGISTIC))) {
            throw std::runtime_error("SigmoidLayer: Validation failed");
          }

          act_.configure(
              &input, &output,
              ActivationLayerInfo(
                  ActivationLayerInfo::ActivationFunction::LOGISTIC));
          break;
        }
        case ElementwiseOp::kSwish: {
          if (!NEActivationLayer::validate(
                  input.info(), output.info(),
                  ActivationLayerInfo(
                      ActivationLayerInfo::ActivationFunction::SWISH))) {
            throw std::runtime_error("SwishLayer: Validation failed");
          }

          act_.configure(&input, &output,
                         ActivationLayerInfo(
                             ActivationLayerInfo::ActivationFunction::SWISH));
          break;
        }
        default:
          throw std::runtime_error(
              "ElementwiseLayer: This operation requires two inputs");
      }
      configured_ = true;
    } catch (const std::exception& e) {
      configured_ = false;
      std::cerr << "ElementwiseLayer configuration error: " << e.what()
                << std::endl;
    }
  }

  void configure(const TensorShape& input1_shape,
                 const TensorShape& input2_shape, TensorShape& output_shape,
                 Tensor& input1, Tensor& input2, Tensor& output) {
    try {
      input1.allocator()->init(TensorInfo(input1_shape, 1, DataType::F32));
      input2.allocator()->init(TensorInfo(input2_shape, 1, DataType::F32));
      output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

      input1.allocator()->allocate();
      input2.allocator()->allocate();
      output.allocator()->allocate();
      switch (op_type_) {
        case ElementwiseOp::kAdd: {
          if (!NEArithmeticAddition::validate(input1.info(), input2.info(),
                                              output.info(),
                                              ConvertPolicy::WRAP)) {
            throw std::runtime_error("AddLayer: Validation failed");
          }

          add_.configure(&input1, &input2, &output, ConvertPolicy::WRAP);
          break;
        }
        case ElementwiseOp::kDiv: {
          if (!NEElementwiseDivision::validate(input1.info(), input2.info(),
                                               output.info())) {
            throw std::runtime_error("DivLayer: Validation failed");
          }

          div_.configure(&input1, &input2, &output);
          break;
        }
        case ElementwiseOp::kSquaredDiff: {
          if (!NEElementwiseSquaredDiff::validate(input1.info(), input2.info(),
                                                  output.info())) {
            throw std::runtime_error("SquaredDiffLayer: Validation failed");
          }

          sqdiff_.configure(&input1, &input2, &output);
          break;
        }
        default:
          throw std::runtime_error(
              "ElementwiseLayer: This operation requires single input");
      }
      configured_ = true;
    } catch (const std::exception& e) {
      configured_ = false;
      std::cerr << "ElementwiseLayer configuration error: " << e.what()
                << std::endl;
    }
  }

  void exec() override {
    if (!configured_) {
      throw std::runtime_error(
          "ElementwiseLayer: Layer not configured before exec.");
    }
    switch (op_type_) {
      case ElementwiseOp::kAbs:
      case ElementwiseOp::kSigm:
      case ElementwiseOp::kSwish:
        act_.run();
        break;
      case ElementwiseOp::kAdd: {
        add_.run();
        break;
      }
      case ElementwiseOp::kDiv: {
        div_.run();
        break;
      }
      case ElementwiseOp::kSquaredDiff: {
        sqdiff_.run();
        break;
      }
      default:
        throw std::runtime_error(
            "ElementwiseLayer: This operation requires single input");
    }
  }

  std::string get_type_name() const override {
    switch (op_type_) {
      case ElementwiseOp::kAdd:
        return "ElementwiseAddLayer";
      case ElementwiseOp::kDiv:
        return "ElementwiseDivLayer";
      case ElementwiseOp::kAbs:
        return "ElementwiseAbsLayer";
      case ElementwiseOp::kSigm:
        return "ElementwiseSigmoidLayer";
      case ElementwiseOp::kSwish:
        return "ElementwiseSwishLayer";
      case ElementwiseOp::kSquaredDiff:
        return "ElementwiseSquaredDiffLayer";
      default:
        return "ElementwiseUnknownLayer";
    }
  }
};

#endif