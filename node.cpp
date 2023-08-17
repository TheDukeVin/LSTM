
#include "lstm.h"

Data::Data(int size_){
    size = size_;
    data = new double[size];
    gradient = new double[size];
    for(int i=0; i<size; i++){
        data[i] = gradient[i] = 0;
    }
}

Data::Data(int size_, double* data_, double* gradient_){
    size = size_;
    data = data_;
    gradient = gradient_;
}

void Data::resetGradient(){
    for(int i=0; i<size; i++){
        gradient[i] = 0;
    }
}

Node::Node(Data* i1_, Data* i2_, Data* o_){
    i1 = i1_; i2 = i2_; o = o_;
}

UnitaryNode::UnitaryNode(Data* i1_, Data* o_, string op){
    i1 = i1_; o = o_; operation = op;
}

double sigmoid(double x){
    return 1/(1 + exp(-x));
}

double tanh(double x){
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double UnitaryNode::nonlinear(double x){
    if(operation == "sigmoid"){
        return sigmoid(x);
    }
    if(operation == "tanh"){
        return tanh(x);
    }
    assert(operation == "identity");
    return x;
}

double UnitaryNode::dnonlinear(double x){
    if(operation == "sigmoid"){
        double s = sigmoid(x);
        return s * (1 - s);
    }
    if(operation == "tanh"){
        double t = tanh(x);
        return 1 - t * t;
    }
    assert(operation == "identity");
    return 1;
}

// Define forward and backward pass operations

void ConcatNode::forwardPass(){
    for(int i=0; i<i1->size; i++){
        o->data[i] = i1->data[i];
    }
    for(int i=0; i<i2->size; i++){
        o->data[i + i1->size] = i2->data[i];
    }
}

void ConcatNode::backwardPass(){
    for(int i=0; i<i1->size; i++){
        i1->gradient[i] += o->gradient[i];
    }
    for(int i=0; i<i2->size; i++){
        i2->gradient[i] += o->gradient[i + i1->size];
    }
}

void AdditionNode::forwardPass(){
    for(int i=0; i<o->size; i++){
        o->data[i] = i1->data[i] + i2->data[i];
    }
}

void AdditionNode::backwardPass(){
    for(int i=0; i<o->size; i++){
        i1->gradient[i] += o->gradient[i];
        i2->gradient[i] += o->gradient[i];
    }
}

void MultiplicationNode::forwardPass(){
    for(int i=0; i<o->size; i++){
        o->data[i] = i1->data[i] * i2->data[i];
    }
}

void MultiplicationNode::backwardPass(){
    for(int i=0; i<o->size; i++){
        i1->gradient[i] += o->gradient[i] * i2->data[i];
        i2->gradient[i] += o->gradient[i] * i1->data[i];
    }
}

void MatMulNode::forwardPass(){
    for(int i=0; i<o->size; i++){
        double sum = 0;
        for(int j=0; j<i2->size; j++){
            sum += i1->data[i*i2->size + j] * i2->data[j];
        }
        o->data[i] = sum;
    }
}

void MatMulNode::backwardPass(){
    for(int j=0; j<i2->size; j++){
        double sum = 0;
        for(int i=0; i<o->size; i++){
            i1->gradient[i*i2->size + j] += i2->data[j] * o->gradient[i];
            sum += i1->data[i*i2->size + j] * o->gradient[i];
        }
        i2->gradient[j] += sum;
    }
}

void UnitaryNode::forwardPass(){
    for(int i=0; i<o->size; i++){
        o->data[i] = nonlinear(i1->data[i]);
    }
}

void UnitaryNode::backwardPass(){
    for(int i=0; i<o->size; i++){
        i1->gradient[i] += dnonlinear(i1->data[i]) * o->gradient[i];
    }
}
