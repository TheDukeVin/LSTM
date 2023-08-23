
#include "lstm.h"

#ifndef test_h
#define test_h

class GradientTest{
public:
    ModelSeq seq;
    int T;

    GradientTest(){
        Model m(5);
        m.addLSTM(6);
        m.addLSTM(7);
        m.addDense(5);
        m.addOutput(4);
        T = 5;
        seq = ModelSeq(m, T, 1);
    }

    void generateRandom(){
        for(int i=0; i<T; i++){
            for(int j=0; j<seq.paramStore.inputSize; j++){
                seq.inputs[i].data[j] = 2 * (double) rand() / RAND_MAX - 1;
            }
            for(int j=0; j<seq.paramStore.outputSize; j++){
                seq.expectedOutputs[i][j] = 2 * (double) rand() / RAND_MAX - 1;
                seq.validOutput[i][j] = rand() % 2;
            }
        }
    }

    void test(){
        generateRandom();
        seq.paramStore.resetGradient();
        seq.forwardPass();
        seq.backwardPass();
        double initLoss = seq.getLoss();
        double epsilon = 0.00001;
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
};

class Supervised{
public:
    ModelSeq seq;

    // TEST: copy previous input

    // const int size = 3;
    // const int T = 20;

    // Supervised(){
    //     Model m(size);
    //     m.addLSTM(5);
    //     m.addOutput(size);
    //     seq = ModelSeq(m, T);
    // }

    // void generateData(){
    //     int prevIndex = -1;
    //     for(int t=0; t<T; t++){
    //         int index = rand() % size;
    //         for(int i=0; i<size; i++){
    //             seq.inputs[t].data[i] = 0;
    //             seq.expectedOutputs[t][i] = 0;
    //             seq.validOutput[t][i] = true;
    //         }
    //         seq.inputs[t].data[index] = 1;
    //         if(prevIndex != -1) seq.expectedOutputs[t][prevIndex] = 1;
    //         prevIndex = index;
    //     }
    // }

    // TEST: binary addition

    // initParam 0.1
    // learnRate 0.02
    // momentum 0.7
    // batchSize 60

    // Run for 10^7 iterations


    const int size = 3;
    const int T = 20;

    double initParam = 0.1;
    double learnRate = 0.02;
    double momentum = 0.9;
    int batchSize = 60;

    Supervised(){
        Model m(size);
        m.addLSTM(10);
        m.addOutput(size);
        seq = ModelSeq(m, T, initParam);
    }

    void generateData(){
        int sum = 0;
        for(int t=0; t<T; t++){
            int num = rand() % (1 << size);
            sum += num;
            for(int i=0; i<size; i++){
                seq.inputs[t].data[i] = (num / (1 << i)) % 2;
                seq.expectedOutputs[t][i] = (sum / (1 << i)) % 2;
                seq.validOutput[t][i] = true;
            }
        }
    }

    void logExampleData(){
        generateData();
        seq.forwardPass();
        cout<<"Example case\n";
        for(int t=0; t<T; t++){
            for(int i=0; i<seq.paramStore.inputSize; i++){
                cout<<seq.inputs[t].data[i]<<' ';
            }
            cout<<"| ";
            for(int i=0; i<seq.paramStore.outputSize; i++){
                cout<<seq.expectedOutputs[t][i]<<' ';
            }
            cout<<"| ";
            for(int i=0; i<seq.paramStore.outputSize; i++){
                cout<<seq.outputs[t].data[i]<<' ';
            }
            cout<<'\n';
        }
    }

    double finalLoss;
    double finalAcc;
    ofstream fout;

    void train(){
        double lossSum = 0;
        double accSum = 0;
        int evalPeriod = 100000;
        for(int iter=0; iter<10000000; iter++){
            generateData();
            seq.forwardPass();
            seq.backwardPass();
            lossSum += seq.getLoss();
            accSum += accuracy();
            if(iter % batchSize == 0){
                seq.paramStore.updateParams(learnRate / batchSize, momentum);
            }
            if(iter % evalPeriod == 0 && iter > 0){
                fout<<"Iter: "<<iter<< " Loss: "<<(lossSum / evalPeriod)<<" Accuracy: "<<(accSum / evalPeriod)<<'\n';
                if(lossSum / evalPeriod < 5){
                    setLearnRate(0.005);
                }
                // if(lossSum / evalPeriod < 4.5){
                //     setLearnRate(0.002);
                // }
                lossSum = 0;
                accSum = 0;
            }
        }
        finalLoss = lossSum / evalPeriod;
        finalAcc = accSum / evalPeriod;
    }

    void setLearnRate(double lr){
        if(learnRate > lr){
            learnRate = lr;
            fout<<"Learn Rate set to "<<lr<<'\n';
        }
    }

    double accuracy(){ // for binary output
        double correct = 0;
        int count = 0;
        for(int t=0; t<T; t++){
            for(int i=0; i<seq.paramStore.outputSize; i++){
                if(seq.validOutput[t][i]){
                    correct += abs(seq.outputs[t].data[i] - seq.expectedOutputs[t][i]) < 
                               abs(seq.outputs[t].data[i] - (1 - seq.expectedOutputs[t][i]));
                    count ++;
                }
            }
        }
        return correct / count;
    }
};

#endif