/**
 * @file hdnaSplice.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief HDNA Splice example with Base level encoder
 * @version 0.1
 * @date 2024-06-07
 * 
 * @copyright GPL3
 * 
 */

#include <syclhd.hpp>
#include <readExeInputs.hpp>
#include <ResultsHandler.hpp>
#include "readDataset.hpp"

#define ACCURATE_TIME

// DNA Sequence Length in the splice Dataset 
#define SEQ_LENGTH 60

#define SHOW_FIT false
#define RT_METHOD hd::voicehd

int main(int argc, char **argv) {

    // Setting up results handler, reading input data
    ResultsHandler results_handler("HDNA Splice", readExeInputs(argc, argv));

    // Reading datasets, automaticly shuffles and splits
    auto data = hdna::readSpliceDataset(
        PROJECT_PATH_CMAKE "/examples/hdna/datasets/splice.data"
    );

    // Setting up model
    auto mymodel = hd::MAP(results_handler.vector_size);
    if (results_handler.host) mymodel.setQueue(cl::sycl::queue{cl::sycl::host_selector{}});
    results_handler.autoset(mymodel);

    // Encoding atomic Hypervectors
    std::vector<char> dna_bases = 
        {'A', 'C', 'G', 'T'};
    
    results_handler.startGen();
    auto posMat = mymodel.genUnlabeledAtomicRep(SEQ_LENGTH);
    auto DNAMat = mymodel.genAtomicRep(dna_bases);
    results_handler.stopGen();

    // Training
    results_handler.startTraining();
    auto AM = posMat
        .baseLevel(DNAMat,data.first.genes_data)
        .train(data.first.labels,results_handler.rt_steps,results_handler.alpha,RT_METHOD,SHOW_FIT);
    results_handler.stopTraining();

    // Testing
    results_handler.startTesting();
    auto testMat = posMat
        .baseLevel(DNAMat,data.second.genes_data);
    results_handler.success_rate = AM
        .test(testMat, data.second.labels) * 100;
    results_handler.stopTesting();  

    // RESULTS
    results_handler.printToTerminal();
    results_handler.printToFile();
    return 0;

}