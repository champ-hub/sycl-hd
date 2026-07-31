/**
 * @file emg.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief EMG with HDCC Datasets
 * @version 0.1
 * @date 2024-05-16
 * 
 * @copyright GPL3
 * 
 */

#include <syclhd.hpp>
#include <readExeInputs.hpp>
#include <ResultsHandler.hpp>
#include "readData.hpp"

#define SIGNAL_LEVELS 21
#define CHANNELS 1024

template<typename Tp>
hd::matrix<Tp> encode(hd::matrix<Tp> &Bases, hd::matrix<Tp> &Levels, hd::index_2d_buffer &data, const int ngram = 3){
    return Bases.ngramBaseLevel(Levels,data,ngram);
    //return Bases.baseLevel(Levels,data);
}


int main(int argc, char **argv) {

    // Setting up results handler, reading input data
    ResultsHandler results_handler("EMG", readExeInputs(argc, argv));

    // Setting up model
    auto model = hd::MAP(results_handler.vector_size);
    if (results_handler.host) model.setQueue(cl::sycl::queue{cl::sycl::host_selector{}});


    results_handler.autoset(model);
    
    results_handler.startGen();
    auto signals = model.genUnlabeledAtomicRep(SIGNAL_LEVELS,hd::full_level);
    auto channels = model.genUnlabeledAtomicRep(CHANNELS);
    results_handler.stopGen();

    std::vector<float> accuracy(5,0);

    for (int i = 1; i <= 5; i++) {
        auto s = std::to_string(i);
        std::cout << "============================"   "="   "=======" << std::endl;
        std::cout << "==== USING DATASET SUBJECT: " << s << " ======" << std::endl;
        
        auto dataset = emg::readDataset(i);
        
        results_handler.startTraining();
        auto AM = encode(channels,signals,dataset.train_data)
            .train(dataset.train_labels);
        results_handler.stopTraining();

        results_handler.startTesting();
        auto entries = encode(channels, signals, dataset.test_data);
        results_handler.success_rate = 
            AM.test(entries,dataset.test_labels)*100;
        results_handler.stopTesting();

        results_handler.example_name = "EMG " + s;
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

