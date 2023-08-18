
#include "lstm.h"

void test(){
    // Build model
    Model m(5);
    m.addLSTM(6);
    m.addOutput(4);
    int T = 3;
    ModelSeq seq(m, T);
    for(int i=0; i<T; i++){
        for(int j=0; j<seq.paramStore.inputSize; j++){
            seq.inputs[i].data[j] = 2 * (double) rand() / RAND_MAX - 1;
        }
        for(int j=0; j<seq.paramStore.outputSize; j++){
            seq.expectedOutputs[i][j] = 2 * (double) rand() / RAND_MAX - 1;
            seq.validOutput[i][j] = rand() % 2;
        }
        seq.forwardPassUnit(i);
    }
    double initLoss = seq.getLoss();
    seq.backwardPass();

    double epsilon = 0.0001;
    double tol = 0.001;
    for(int i=0; i<seq.paramStore.layers.size(); i++){
        for(int j=0; j<seq.paramStore.layers[i]->params.size; j++){
            seq.paramStore.layers[i]->params.params[j] += epsilon;
            seq.forwardPass();
            double newLoss = seq.getLoss();
            seq.paramStore.layers[i]->params.params[j] -= epsilon;
            double derivative = (newLoss - initLoss) / epsilon;
            // cout<<derivative<<' '<<seq.paramStore.layers[i]->params.gradient[j]<<'\n';
            assert(abs(derivative - seq.paramStore.layers[i]->params.gradient[j]) < tol);
        }
    }
}

int main(){
    for(int i=0; i<100; i++){
        test();
    }
}