#ifndef ACL_SPLIT_LAYER_H
#define ACL_SPLIT_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

class SplitLayer : public Layer {
private:
    NESplit split;
    bool configured_ = false;

public:
    SplitLayer(int id) { setID(id); }

    void configure(const TensorShape& input_shape, unsigned int axis, Tensor& input, std::vector<ITensor*>& outputs) {
        input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
        input.allocator()->allocate();

        split.configure(&input, outputs, axis);
        configured_ = true;
    }

    void exec() override {
        if (!configured_) {
            throw std::runtime_error("SplitLayer: Layer not configured.");
        }
        split.run();
    }

    std::string get_type_name() const override { return "SplitLayer"; }
};

#endif