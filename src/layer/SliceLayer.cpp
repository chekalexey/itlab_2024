#ifndef ACL_SLICE_LAYER_H
#define ACL_SLICE_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

class SliceLayer : public Layer {   
private:
    NESlice slice;
    bool configured_ = false;

public:
    SliceLayer(int id) { setID(id); }

    void configure(const TensorShape& input_shape, Coordinates starts, Coordinates ends, 
        TensorShape& output_shape, Tensor& input, Tensor& output) {

        input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
        output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

        input.allocator()->allocate();
        output.allocator()->allocate();

        slice.configure(&input, &output, starts, ends);
        configured_ = true;
    }

    void exec() override {
        if (!configured_) {
            throw std::runtime_error("SliceLayer: Layer not configured.");
        }
        slice.run();
    }
    std::string get_type_name() const override { return "SliceLayer"; }
};

#endif