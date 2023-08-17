
#include "lstm.h"
#define numInputs 5
#define numOutputs 15

class LSTMSeq{
public:
    int T; // length of LSTM sequence
    LSTMParams params;
    LSTM* units;
    Data* inputs;
    Data* outputs;
    double** expectedOutputs;
    // vector<LSTM> units;
    // vector<Data> inputs;
    // vector<Data> outputs;
    // vector<double*> expectedOutputs;

    LSTMSeq(int T_){
        T = T_;
        units = new LSTM[T];
        inputs = new Data[T];
        outputs = new Data[T];
        expectedOutputs = new double*[T];
        params = LSTMParams(numInputs, numOutputs);
        params.randomize();
        for(int i=0; i<T; i++){
            inputs[i] = Data(numInputs);
            outputs[i] = Data(numOutputs);
            expectedOutputs[i] = new double[numOutputs];
        }
        for(int i=0; i<T; i++){
            LSTM* prevUnit;
            if(i > 0){
                prevUnit = &units[i-1];
            }
            else{
                prevUnit = NULL;
            }
            units[i] = LSTM(&inputs[i], &outputs[i], prevUnit, params);
        }
    }

    double FBpass(){ // returns loss
        for(int i=0; i<T; i++){
            units[i].params.copy(params);
            units[i].forwardPass();
        }
        double sum = 0;
        for(int i=T-1; i>=0; i--){
            for(int j=0; j<numOutputs; j++){
                if(expectedOutputs[i][j] == -2){
                    units[i].output->gradient[j] = 0;
                    continue;
                }
                units[i].output->gradient[j] = 2 * (units[i].output->data[j] - expectedOutputs[i][j]);
                sum += pow(units[i].output->data[j] - expectedOutputs[i][j], 2);
            }
            units[i].backwardPass();
            params.accumulateGradient(units[i].params);
        }
        return sum;
    }
};

void test(){
    LSTMSeq seq(5);
    for(int i=0; i<seq.T; i++){
        for(int j=0; j<numInputs; j++){
            seq.inputs[i].data[j] = 2 * (double) rand() / RAND_MAX - 1;
        }
        for(int j=0; j<numOutputs; j++){
            if(rand() % 2 == 0){
                seq.expectedOutputs[i][j] = -2;
                continue;
            }
            seq.expectedOutputs[i][j] = 2 * (double) rand() / RAND_MAX - 1;
        }
    }

    double initLoss = seq.FBpass();
    LSTMParams save(numInputs, numOutputs);
    save.copy(seq.params);

    double epsilon = 0.0001;
    for(int p=0; p<seq.params.size; p++){
        seq.params.params[p] += epsilon;
        double newLoss = seq.FBpass();
        seq.params.params[p] -= epsilon;
        double derivative = (newLoss - initLoss) / epsilon;
        // cout<<derivative<<' '<<save.gradient[p]<<'\n';
        assert(abs(derivative - save.gradient[p]) < 0.001);
    }
}

// TEST: Simple test of outputting increasing sequence

// Run for 300000 iterations
// Train at 20 time steps

// numInputs 1
// numOutputs 3

// LR 0.1

// void inputData(){
//     for(int i=0; i<T; i++){
//         seq.inputs[i].data[0] = 0;
//         for(int j=0; j<numOutputs; j++){
//             seq.expectedOutputs[i][j] = -2;
//         }
//         seq.expectedOutputs[i][0] = (double) i / T;
//     }
// }



// TEST: Saves and outputs previous value

// Run for 300000 iterations
// Train at 20 time steps

// numInputs 3
// numOutputs 6

// LR 0.1

// void inputData(LSTMSeq* seq){
//     int prevValue = -1;
//     for(int i=0; i<seq->T; i++){
//         int value = rand() % numInputs;
//         for(int j=0; j<numInputs; j++){
//             seq->inputs[i].data[j] = 0;
//         }
//         seq->inputs[i].data[value] = 1;
//         for(int j=0; j<numOutputs; j++){
//             seq->expectedOutputs[i][j] = -2;
//         }
//         for(int j=0; j<numInputs; j++){
//             seq->expectedOutputs[i][j] = 0;
//         }
//         if(prevValue != -1) seq->expectedOutputs[i][prevValue] = 1;
//         prevValue = value;
//     }
// }

// TEST: Saves/outputs values from 1,2,4,8,16 time steps ago

// Run for 10^6 iterations
// Train at 100 time steps

// numInputs 5
// numOutputs 15

// LR 0.005
// Momentum 0.7
// Batch 10

void inputData(LSTMSeq* seq){
    int counts[seq->T];
    for(int i=0; i<seq->T; i++){
        for(int j=0; j<numOutputs; j++){
            seq->expectedOutputs[i][j] = -2;
        }
        for(int j=0; j<numInputs; j++){
            seq->inputs[i].data[j] = 0;
            seq->expectedOutputs[i][j] = 0;
        }
        counts[i] = 0;
    }
    for(int i=0; i<seq->T; i++){
        int value = rand() % numInputs;
        seq->inputs[i].data[value] = 1;
        for(int j=0; j<numInputs; j++){
            if(j < counts[i]){
                seq->expectedOutputs[i][j] = 1;
            }
        }
        int newIndex = i + (1 << value);
        if(newIndex < seq->T){
            counts[newIndex] ++;
        }
    }
}

double accuracy(LSTMSeq* seq){ // for binary 0/1 output data
    double correct = 0;
    int count = 0;
    for(int i=0; i<seq->T; i++){
        for(int j=0; j<numOutputs; j++){
            if(seq->expectedOutputs[i][j] == -2) continue;
            correct += abs(seq->outputs[i].data[j] - seq->expectedOutputs[i][j]) < 
                       abs(seq->outputs[i].data[j] - (1 - seq->expectedOutputs[i][j]));
            count ++;
        }
    }
    return correct / count;
}

void train(LSTMSeq* seq){
    double sum = 0;
    double sumAcc = 0;
    int evalPeriod = 100000;
    for(int i=0; i<1000000; i++){
        if(i > 0 && i % evalPeriod == 0){
            cout<<"Loss: "<<(sum / evalPeriod)<<'\n';
            sum = 0;
            cout<<"Accuracy: "<<(sumAcc / evalPeriod) << '\n';
            sumAcc = 0;
        }
        inputData(seq);
        sum += seq->FBpass();
        sumAcc += accuracy(seq);
        if(i%batchSize == 0){
            seq->params.update();
        }
    }
}

void extend(LSTMSeq* seq, int newLength){
    LSTMSeq newSeq(newLength);
    newSeq.params.copy(seq->params);
    double sum = 0;
    double sumAcc = 0;
    int numEval = 100;
    for(int i=0; i<numEval; i++){
        inputData(&newSeq);
        sum += newSeq.FBpass();
        sumAcc += accuracy(&newSeq);
    }
    cout<<"Extended loss: "<<(sum / numEval)<<'\n';
    cout<<"Extended accuracy: "<<(sumAcc / numEval)<<'\n';
}

void logExampleData(LSTMSeq* seq){
    inputData(seq);
    for(int i=0; i<seq->T; i++){
        for(int j=0; j<numInputs; j++){
            cout<<seq->inputs[i].data[j]<<' ';
        }
        cout<<"| ";
        for(int j=0; j<numOutputs; j++){
            cout<<seq->expectedOutputs[i][j]<<' ';
        }
        cout<<'\n';
    }
}

int main(){
    // for(int i=0; i<100; i++){
    //     test();
    // }

    LSTMSeq seq(100);
    // logExampleData(&seq);
    train(&seq);
    extend(&seq, 300);
}