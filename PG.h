
/*

g++ -O2 -std=c++17 -pthread -I /Users/kevindu/Desktop/Coding/Tests:experiments/LSTM main.cpp modelseq.cpp model.cpp layer.cpp lstm.cpp policy.cpp params.cpp node.cpp PG.cpp common.cpp dice/environment.cpp

g++ -O2 -std=c++17 -pthread -I /n/home04/kevindu/MultiagentSnake/LSTM main.cpp modelseq.cpp model.cpp layer.cpp lstm.cpp policy.cpp params.cpp node.cpp PG.cpp common.cpp token/environment.cpp

*/

#include "lstm.h"
// #include "token/environment.h"
#include "dice/environment.h"

#ifndef PG_h
#define PG_h

class PG{
public:
    ModelSeq seq;

    double reward[TIME_HORIZON];

    double valueMean[TIME_HORIZON];

    double initParam = 0.1;
    double learnRate = 0.005;
    double momentum = 0;
    int batchSize = 30;
    double meanUpdate = 0.0001;

    PG(){
        Model m(inputSize);
        m.addLSTM(10);
        m.addOutput(NUM_ACTIONS);
        seq = ModelSeq(m, TIME_HORIZON, initParam);
    }

    ofstream fout;
    double finalReward;

    double rollOut(bool printGame = false); // returns cumulative reward
    void computeSoftmax(double* weights, double* policy, vector<int> validActions);
    void train();
    void setLearnRate(double lr);
};

#endif