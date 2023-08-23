
#include "lstm.h"
// #include "test.h"
#include "PG.h"

// Run training sessions in parallel

// const int numThreads = 10;

// Supervised trainers[numThreads];
// thread* threads[numThreads];

// void runThread(int i){
//     trainers[i].fout = ofstream("session" + to_string(i) + ".out");
//     trainers[i].train();
// }

// void testGrad(){
//     GradientTest grad;
//     for(int i=0; i<100; i++){
//         grad.test();
//     }
// }

// void testSuper(){
//     for(int i=0; i<numThreads; i++){
//         threads[i] = new thread(runThread, i);
//     }
//     for(int i=0; i<numThreads; i++){
//         threads[i]->join();
//         cout<<trainers[i].finalLoss<<' '<<trainers[i].finalAcc<<'\n';
//     }

//     // Supervised super;
//     // super.train();
// }

const int numThreads = 1;
PG trainers[numThreads];

void runThread(int i){
    trainers[i].fout = ofstream("session" + to_string(i) + ".out");
    trainers[i].train();
}

int main(){
    srand(time(0));
    int start_time = time(0);

    // testGrad();

    // testSuper();

    // thread* threads[numThreads];
    // for(int i=0; i<numThreads; i++){
    //     threads[i] = new thread(runThread, i);
    // }
    // for(int i=0; i<numThreads; i++){
    //     threads[i]->join();
    //     cout<<trainers[i].finalReward<<'\n';
    // }

    trainers[0].train();
    // for(int i=0; i<10; i++){
    //     trainers[0].rollOut(true);
    // }
    

    cout<<"TIME: "<<(time(0) - start_time)<<'\n';
}