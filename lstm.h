
/*
g++ -O2 -std=c++11 main.cpp node.cpp lstm.cpp
*/

#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <math.h>

#ifndef lstm_h
#define lstm_h
using namespace std;

#define initParam 0.01
#define learnRate 0.005
#define momentum 0.7
#define batchSize 10

class Data{
public:
    int size;
    double* data;
    double* gradient;

    Data(){}
    Data(int size_);
    Data(int size_, double* data_, double* gradient_);

    void resetGradient();
};

class Node{
public:
    Data* i1;
    Data* i2;
    Data* o;

    Node(){}
    Node(Data* i1_, Data* i2_, Data* o_);

    virtual void forwardPass() = 0;
    virtual void backwardPass() = 0;
};

class ConcatNode : public Node{
public:
    using Node::Node;
    void forwardPass();
    void backwardPass();
};

class AdditionNode : public Node{
public:
    using Node::Node;
    void forwardPass();
    void backwardPass();
};

class MultiplicationNode : public Node{
public:
    using Node::Node;
    void forwardPass();
    void backwardPass();
};

class MatMulNode : public Node{ // (m x n) matrix times (n x 1) vector -> (m x 1) vector
public:
    using Node::Node;
    void forwardPass();
    void backwardPass();
};

class UnitaryNode : public Node{
public:
    string operation;

    UnitaryNode(Data* i1_, Data* o_, string op);
    
    void forwardPass();
    void backwardPass();

    double nonlinear(double x);
    double dnonlinear(double x);
};

class LSTMParams{
public:
    int size;
    double* params;
    double* gradient;

    LSTMParams(){}
    LSTMParams(int inputSize, int outputSize);
    void randomize();
    void copy(LSTMParams params_);
    void accumulateGradient(LSTMParams params_);
    void update();
    void resetGradient();
};

class LSTM{
public:
    LSTMParams params;

    int inputSize;
    int outputSize;

    Data* input;
    Data* hidden;
    Data* cell;
    Data* output;

    vector<Data*> allHiddenData;
    vector<Node*> allNodes;

    LSTM(){}
    LSTM(int size); // empty LSTM
    LSTM(Data* input_, Data* output_, LSTM* prevUnit, LSTMParams params_);
    Data* addData(int size);

    void forwardPass(); // resets gradient of all hidden data
    void backwardPass();
    void resetGradient();
};

#endif