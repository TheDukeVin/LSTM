
#include "lstm.h"

LSTMParams::LSTMParams(int inputSize, int outputSize){
    size = 4 * ((inputSize + outputSize + 1) * outputSize);
    params = new double[size];
    gradient = new double[size];
    for(int i=0; i<size; i++){
        params[i] = gradient[i] = 0;
    }
}

void LSTMParams::randomize(){
    for(int i=0; i<size; i++){
        params[i] = (2 * (double) rand() / RAND_MAX - 1) * initParam;
    }
}

void LSTMParams::copy(LSTMParams params_){
    for(int i=0; i<size; i++){
        params[i] = params_.params[i];
        gradient[i] = params_.gradient[i];
    }
}

void LSTMParams::accumulateGradient(LSTMParams params_){
    for(int i=0; i<size; i++){
        gradient[i] += params_.gradient[i];
    }
}

void LSTMParams::update(){
    for(int i=0; i<size; i++){
        params[i] -= gradient[i] * learnRate / batchSize;
        gradient[i] *= momentum;
        assert(abs(params[i]) < 1000);
    }
}

void LSTMParams::resetGradient(){
    for(int i=0; i<size; i++){
        gradient[i] = 0;
    }
}

LSTM::LSTM(int size){
    hidden = new Data(size);
    cell = new Data(size);
    for(int i=0; i<size; i++){
        hidden->data[i] = hidden->gradient[i] = 0;
        cell->data[i] = cell->gradient[i] = 0;
    }
}

LSTM::LSTM(Data* input_, Data* output_, LSTM* prevUnit, LSTMParams params_){
    input = input_;
    output = output_;
    inputSize = input->size;
    outputSize = output->size;
    params = LSTMParams(inputSize, outputSize);
    params.copy(params_);
    if(prevUnit == NULL){
        prevUnit = new LSTM(outputSize);
    }

    // Initialize Data and Nodes

    Data* XH = addData(inputSize + outputSize);
    allNodes.push_back(new ConcatNode(input, prevUnit->hidden, XH));

    int weightSize = (inputSize + outputSize) * outputSize;
    int biasSize = outputSize;

    Data* weights[4];
    Data* bias[4];
    vector<Data*> linComb;
    for(int i=0; i<4; i++){
        weights[i] = new Data(weightSize, params.params + (weightSize + biasSize)*i,
                                          params.gradient + (weightSize + biasSize)*i);
        bias[i] = new Data(biasSize, params.params + (weightSize + biasSize)*i + weightSize,
                                     params.gradient + (weightSize + biasSize)*i + weightSize);
        Data* mult_result = addData(outputSize);
        allNodes.push_back(new MatMulNode(weights[i], XH, mult_result));
        Data* sum_result = addData(outputSize);
        allNodes.push_back(new AdditionNode(mult_result, bias[i], sum_result));
        linComb.push_back(sum_result);
    }
    Data* F = addData(outputSize);
    allNodes.push_back(new UnitaryNode(linComb[0], F, "sigmoid"));
    Data* G = addData(outputSize);
    allNodes.push_back(new UnitaryNode(linComb[1], G, "sigmoid"));
    Data* H = addData(outputSize);
    allNodes.push_back(new UnitaryNode(linComb[2], H, "sigmoid"));
    Data* C = addData(outputSize);
    allNodes.push_back(new UnitaryNode(linComb[3], C, "tanh"));

    Data* C1 = addData(outputSize);
    allNodes.push_back(new MultiplicationNode(F, prevUnit->cell, C1));
    Data* C2 = addData(outputSize);
    allNodes.push_back(new MultiplicationNode(G, C, C2));
    cell = addData(outputSize);
    allNodes.push_back(new AdditionNode(C1, C2, cell));

    Data* feedback = addData(outputSize);
    allNodes.push_back(new UnitaryNode(cell, feedback, "tanh"));
    hidden = addData(outputSize);
    allNodes.push_back(new MultiplicationNode(H, feedback, hidden));

    allNodes.push_back(new UnitaryNode(hidden, output, "identity"));
}

Data* LSTM::addData(int size){
    Data* data = new Data(size);
    allHiddenData.push_back(data);
    return data;
}

void LSTM::forwardPass(){
    resetGradient();
    for(int i=0; i<allNodes.size(); i++){
        allNodes[i]->forwardPass();
    }
}

void LSTM::backwardPass(){
    for(int i=allNodes.size()-1; i>=0; i--){
        allNodes[i]->backwardPass();
    }
}

void LSTM::resetGradient(){
    input->resetGradient();
    hidden->resetGradient();
    cell->resetGradient();
    output->resetGradient();
    for(int i=0; i<allHiddenData.size(); i++){
        allHiddenData[i]->resetGradient();
    }
    params.resetGradient();
}