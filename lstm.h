
/*
g++ -O2 -std=c++11 -pthread main.cpp modelseq.cpp model.cpp layer.cpp lstm.cpp policy.cpp params.cpp node.cpp

-fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fno-sanitize=null -fno-sanitize=alignment

rsync -r LSTM kevindu@login.rc.fas.harvard.edu:./MultiagentSnake

*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <thread>
#include <cassert>

#ifndef lstm_h
#define lstm_h
using namespace std;

// #define initParam 0.1
// #define learnRate 0.05
// #define momentum 0.7
// #define batchSize 60

int sampleDist(double* dist, int N);

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

class Params{
public:
    int size;
    double* params;
    double* gradient;

    Params(){}
    Params(int size_);
    void randomize(double scale);
    void copy(Params params_);
    void accumulateGradient(Params params_);
    void update(double scale, double momentum);
    void resetGradient();
};

class Layer{
protected:
    vector<Data*> allHiddenData;
    vector<Node*> allNodes;

    Data* addData(int size);
    void resetGradient();
    
public:
    Params params;

    int inputSize;
    int outputSize;

    Data* input;
    Data* output;

    Layer(){}

    void forwardPass(); // resets gradient of all data
    void backwardPass();

    virtual void vf(){};
};

class LSTM : public Layer{
public:
    Data* cell;

    // Looks at previous unit's output and cell.
    LSTM(int size); // empty LSTM to start the chain
    LSTM(Data* input_, Data* output_, LSTM* prevUnit);
};

class Dense : public Layer{
protected:
    void setupLayer(Data* input_, Data* output_, string operation);

public:
    Dense(){}
    Dense(Data* input_, Data* output_);
};

class PolicyOutput : public Dense{
public:
    PolicyOutput(Data* input_, Data* output_);
};

class Model{
private:
    int lastSize = -1; // used to initialize model
    Data* lastAct;

public:
    vector<Layer*> layers;
    vector<Data*> activations;

    int inputSize;
    int outputSize;

    Model(){}

    // Construct a model structure
    Model(int inputSize_);
    void addLSTM(int outputSize_);
    void addDense(int outputSize_);
    void addOutput(int outputSize_);

    // Define an active Model unit from given structure
    Model(Model structure, Model* prevModel, Data* input, Data* output);

    void copyParams(Model* m);
    void randomize(double scale);

    void forwardPass();
    void backwardPass();

    void resetGradient();
    void accumulateGradient(Model* m);
    void updateParams(double scale, double momentum);
};

class ModelSeq{
public:
    int T;
    vector<Model> seq;
    vector<Data> inputs;
    vector<Data> outputs;
    // vector<double*> expectedOutputs;
    // vector<bool*> validOutput;
    Model paramStore;

    ModelSeq(){}
    ModelSeq(Model structure, int T_, double initParam);
    void forwardPassUnit(int index);
    void forwardPass();
    void backwardPassUnit(int index);
    void backwardPass();

    double getLoss();
};

#endif