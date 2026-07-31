/**
 * @file hdnaBats.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief HDNA DNA sequencing bats example 
 * Only works for MAP-BP/ MAP due to the existence of "null bases" that are set to zero
 * 
 * @version 0.2
 * @date 2024-01-21
 * 
 * @copyright GPL3
 * 
 */
#include <syclhd.hpp>
#include <readExeInputs.hpp>
#include <ResultsHandler.hpp>
#include "readDataset.hpp"

#define ACCURATE_TIME

//You can set a 'shift_depth' based on this example
//static const hd::permutation p = (hd::permutation) 8;
static const hd::permutation p(hd::shift_left);

int main(int argc, char **argv) {

    // Setting up results handler, reading input data
    ResultsHandler results_handler("HDNA Bats", readExeInputs(argc, argv));

    // Reading datasets
    auto train_data = hdna::readBatsDataset(
            PROJECT_PATH_CMAKE "/examples/hdna/datasets/batsTrain.fas");
    auto test_data = hdna::readBatsDataset(
            PROJECT_PATH_CMAKE "/examples/hdna/datasets/batsTest.fas");


    // Setting up model
    auto mymodel = hd::TMAP(results_handler.vector_size);
    if (results_handler.host) mymodel.setQueue(cl::sycl::queue{cl::sycl::host_selector{}});

    results_handler.autoset(mymodel);
    const int n_gram_sz = 10;


    // Encoding atomic Hypervectors
    std::vector<char> dna_bases = 
        {'A', 'C', 'G', 'T'};

    std::vector<char> null_bases = {
         'N', 'Y', 'M', 'S',
         'a', 'c', 'g', 't',
         'K','W', '-','R',
         'D'};

    results_handler.startGen();
    auto null_bases_rep = mymodel.genAtomicRep(null_bases,hd::all_zero);

    auto dna_bases_rep = 
        mymodel
            .genAtomicRep(dna_bases,hd::random,hd::ignore)
            .stack(null_bases_rep);
    results_handler.stopGen();


    // TRAINING
    results_handler.startTraining();

    auto associative_memory = 
        dna_bases_rep
            .ngram(train_data.genes_data,n_gram_sz,p)
            .train(train_data.labels); //, results_handler.rt_steps);

    results_handler.stopTraining();


    // TESTING
    results_handler.startTesting();
    
    auto encoded_test_entries = 
        dna_bases_rep.ngram(test_data.genes_data,n_gram_sz,p);

    results_handler.success_rate =
            associative_memory.test(encoded_test_entries, test_data.labels) * 100;

    results_handler.stopTesting();


    // RESULTS
    results_handler.printToTerminal();
    results_handler.printToFile();
    return 0;
}