#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "utils/Utils.h"

#include <iostream>
using namespace arm_compute;
using namespace utils;

int main(){
    Tensor input;
    Tensor weight;
    Tensor bias;
    Tensor output;

    const unsigned int Hin = 3;
    const unsigned int Win = 3;
    const unsigned int Cin = 1;

    const unsigned int Hf = 3;
    const unsigned int Wf = 3;

    const unsigned int Hout = Hin - Hf + 1;
    const unsigned int Wout = Win - Wf + 1;
    const unsigned int Cout = 1;

    input.allocator()->init(TensorInfo(TensorShape(Hin, Win, Cin), 1, DataType::F32));
    weight.allocator()->init(TensorInfo(TensorShape(Hf, Wf, Cin, Cout), 1, DataType::F32));
    output.allocator()->init(TensorInfo(TensorShape(Hout, Wout, Cout), 1, DataType::F32));

    input.allocator()->allocate();
    weight.allocator()->allocate();
    output.allocator()->allocate();

    NEConvolutionLayer conv;
    conv.configure(&input, &weight, nullptr, &output, PadStrideInfo(1, 1, 0, 0));

    conv.run();

    output.print(std::cout);
}