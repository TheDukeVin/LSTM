
#include "PG.h"

double PG::rollOut(bool printGame){
    Environment env;

    double policy[TIME_HORIZON][NUM_ACTIONS];
    Environment states[TIME_HORIZON];
    int actions[TIME_HORIZON];

    int rollOutLength = TIME_HORIZON;
    double rewardSum = 0;
    for(int t=0; t<TIME_HORIZON; t++){
        states[t] = env;
        env.inputObservations(&seq.inputs[t]);
        seq.forwardPassUnit(t);
        computeSoftmax(seq.outputs[t].data, policy[t], env.validActions());

        // double sum = 0;
        // for(int i=0; i<NUM_ACTIONS; i++){
        //     policy[t][i] = -1;
        // }
        // for(auto a : env.validActions()){
        //     policy[t][a] = exp(seq.outputs[t].data[a]);
        //     sum += policy[t][a];
        // }
        // for(auto a : env.validActions()){
        //     policy[t][a] /= sum;
        // }
        int action = sampleDist(policy[t], NUM_ACTIONS);
        
        if(printGame){
            cout << "Env: " << env.toString() << '\n';
            cout << "Policy: ";
            for(int i=0; i<NUM_ACTIONS; i++){
                cout << policy[t][i] << ' ';
            }
            cout << '\n';
            cout << "Action: " << action << '\n';
        }

        actions[t] = action;
        reward[t] = env.makeAction(action);
        if(printGame){
            cout << "Reward: " << reward[t] << '\n';
        }
        rewardSum += reward[t];
        if(env.endState){
            rollOutLength = t+1;
            break;
        }
    }
    double value = 0;
    for(int t=rollOutLength-1; t>=0; t--){
        value *= DISCOUNT_FACTOR;
        value += reward[t];
        double scale = value - valueMean[t];
        valueMean[t] += (value - valueMean[t]) * meanUpdate;
        for(auto a : states[t].validActions()){
            seq.outputs[t].gradient[a] = policy[t][a] * scale;
        }
        seq.outputs[t].gradient[actions[t]] = (policy[t][actions[t]] - 1) * scale;
        seq.backwardPassUnit(t);
    }
    return rewardSum;
}

void PG::computeSoftmax(double* weights, double* policy, vector<int> validActions){
    for(int i=0; i<NUM_ACTIONS; i++){
        policy[i] = -1;
    }
    double maxWeight = -1e+08;
    for(auto a : validActions){
        maxWeight = max(maxWeight, weights[a]);
    }
    double sum = 0;
    for(auto a : validActions){
        assert(!isnan(weights[a]));
        policy[a] = exp(weights[a] - maxWeight);
        sum += policy[a];
    }
    for(auto a : validActions){
        policy[a] /= sum;
    }
}

void PG::train(){
    double rewardSum = 0;
    int evalPeriod = 100000;
    for(int i=0; i<TIME_HORIZON; i++){
        valueMean[i] = 0;
    }
    for(int iter=0; iter<3000000; iter++){
        // cout<<"NEW ROLLOUT:\n";
        rewardSum += rollOut();
        if(iter % batchSize == 0){
            seq.paramStore.updateParams(learnRate / batchSize, momentum);
        }
        if(iter % evalPeriod == 0 && iter > 0){
            cout<<"Iter: "<<iter<< ". Average Reward: "<<(rewardSum / evalPeriod)<<'\n';
            // if(rewardSum / evalPeriod > 4){
            //     setLearnRate(0.005);
            // }
            rewardSum = 0;
        }
    }
    finalReward = rewardSum / evalPeriod;
}

void PG::setLearnRate(double lr){
    if(learnRate > lr){
        learnRate = lr;
        fout<<"Learn Rate set to "<<lr<<'\n';
    }
}