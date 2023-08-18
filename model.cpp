
#include "lstm.h"

Model::Model(int inputSize_){
    inputSize = inputSize_;
    lastSize = inputSize_;
    lastAct = new Data(inputSize);
    activations.push_back(lastAct);
}

void Model::addLSTM(int outputSize_){
    Data* newAct = new Data(outputSize_);
    layers.push_back(new LSTM(lastAct, newAct, NULL));
    activations.push_back(newAct);
    lastSize = outputSize_;
    lastAct = newAct;
}

void Model::addOutput(int outputSize_){
    Data* newAct = new Data(outputSize_);
    layers.push_back(new PolicyOutput(lastAct, newAct));
    activations.push_back(newAct);
    outputSize = outputSize_;
    lastAct = newAct;
}

Model::Model(Model structure, Model* prevModel, Data* input, Data* output){
    inputSize = structure.inputSize;
    outputSize = structure.outputSize;
    activations.push_back(input);
    lastAct = input;
    for(int i=0; i<structure.layers.size(); i++){
        Data* newAct;
        if(i == structure.layers.size()-1){
            newAct = output;
        }
        else{
            newAct = new Data(structure.layers[i]->outputSize);
        }
        activations.push_back(newAct);
        if(dynamic_cast<LSTM*>(structure.layers[i]) != NULL){
            LSTM* prevLSTM;
            if(prevModel == NULL) prevLSTM = NULL;
            else prevLSTM = dynamic_cast<LSTM*>(prevModel->layers[i]);
            layers.push_back(new LSTM(lastAct, newAct, prevLSTM));
        }
        else if(dynamic_cast<PolicyOutput*>(structure.layers[i]) != NULL){
            layers.push_back(new PolicyOutput(lastAct, newAct));
        }
        else{
            assert(false);
        }
        lastAct = newAct;
    }
    assert(layers.size() == structure.layers.size());
}

void Model::copyParams(Model* m){
    for(int i=0; i<layers.size(); i++){
        layers[i]->params.copy(m->layers[i]->params);
    }
}

void Model::randomize(){
    for(int i=0; i<layers.size(); i++){
        layers[i]->params.randomize();
    }
}

void Model::forwardPass(){
    for(int i=0; i<layers.size(); i++){
        layers[i]->forwardPass();
    }
}

void Model::backwardPass(){
    for(int i=layers.size()-1; i>=0; i--){
        layers[i]->backwardPass();
    }
}

void Model::accumulateGradient(Model* m){
    for(int i=0; i<layers.size(); i++){
        layers[i]->params.accumulateGradient(m->layers[i]->params);
    }
}