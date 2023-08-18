
#include "lstm.h"

PolicyOutput::PolicyOutput(Data* input_, Data* output_){
    input = input_;
    output = output_;
    inputSize = input->size;
    outputSize = output->size;
    params = Params(inputSize * outputSize + outputSize);

    int weightSize = inputSize * outputSize;
    int biasSize = outputSize;
    Data* weights = new Data(weightSize, params.params, params.gradient);
    Data* bias = new Data(biasSize, params.params + weightSize, params.gradient + weightSize);
    Data* mult_result = addData(outputSize);
    allNodes.push_back(new MatMulNode(weights, input, mult_result));
    allNodes.push_back(new AdditionNode(mult_result, bias, output));
}