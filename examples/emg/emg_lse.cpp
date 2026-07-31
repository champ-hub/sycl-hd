/**
 * @file emg_lse.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-10-17
 * 
 * @copyright GPL3
 * 
 */

#include <syclhd.hpp>
#include <ResultsHandler.hpp>
#include "readData.hpp"

#define SIGNAL_LEVELS 21
#define CHANNELS 1024
#define N_CLASSES 5

template<typename Tp>
hd::matrix<Tp> encode(hd::matrix<Tp> &Bases, hd::matrix<Tp> &Levels, hd::index_2d_buffer &data, const int ngram = 3){
    //return Bases.ngramBaseLevel(Levels,data,ngram);
    return Bases.baseLevel(Levels,data);
}


int main(int argc, char **argv) {

    // Setting up results handler, reading input data
    ResultsHandler results_handler("EMG VSLSC", readExeInputs(argc, argv));
    results_handler.vector_size = 100;

    // Setting up model
    auto model = hd::MAP(results_handler.vector_size);
    if (results_handler.host) model.setQueue(cl::sycl::queue{cl::sycl::host_selector{}});

    results_handler.autoset(model);
    
    results_handler.startGen();
    auto signals = model.genUnlabeledAtomicRep(SIGNAL_LEVELS,hd::full_level);
    auto channels = model.genUnlabeledAtomicRep(CHANNELS);
    results_handler.stopGen();

    hd::lse::classifier c(model.getVectorSize(),N_CLASSES,model.getQueue());
    c.g_i = true; //Make sure corr matrix is invertible 
    
    std::vector<float> accuracy(5,0);

    for (int i = 1; i <= 5; i++) {
        auto s = std::to_string(i);
        std::cout << "============================"   "="   "=======" << std::endl;
        std::cout << "==== USING DATASET SUBJECT: " << s << " ======" << std::endl;
        
        auto dataset = emg::readDataset(i);
        
        auto train_l = hd::index_1d_buffer(dataset.train_labels.begin(), dataset.train_labels.end());
        auto test_l = hd::index_1d_buffer(dataset.test_labels.begin(), dataset.test_labels.end());

        results_handler.startTraining();
        auto train_e = encode(channels,signals,dataset.train_data);
        c.train_weights(train_e,train_l);
        results_handler.stopTraining();

        results_handler.startTesting();
        auto test_e = encode(channels, signals, dataset.test_data);
        results_handler.success_rate = 
            c.test(test_e,test_l)*100;
        results_handler.stopTesting();

        results_handler.example_name = "EMG " + s + " VSLSC";
        results_handler.printToTerminal();
        results_handler.printToFile();
        accuracy[i-1] = results_handler.success_rate;
    }

    std::cout << "===============================" << std::endl;
    std::cout << "======= AVERAGE RESULTS =======" << std::endl << std::endl;
    
    double sum_a = std::accumulate(std::begin(accuracy), std::end(accuracy), 0.0);
    double avg_a =  sum_a / accuracy.size();

    std::cout << "Avg. Accuracy: " << avg_a << "%" << std::endl;

    return 0;
}

