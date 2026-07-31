/**
 * @file hdna_lse.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief HDNA Splice example with Base level encoder and lse classifier
 * @version 0.1
 * @date 2024-10-17
 * 
 * @copyright GPL3
 * 
 */

#include <syclhd.hpp>
#include <ResultsHandler.hpp>
#include "readDataset.hpp"

#define ACCURATE_TIME

// DNA Sequence Length in the splice Dataset 
#define SEQ_LENGTH 60

//#define DIM_DISCOVERY

/*
Dim,Fit
26,66.3962
34,51.5308
35,52.8218
38,25.7101
39,36.3703
43,77.3884
44,24.1977
48,24.3453
50,51.8997
51,51.2726
*/


int main(int argc, char **argv) {

    // Setting up results handler, reading input data
    ResultsHandler results_handler("HDNA Splice VSLSC", readExeInputs(argc, argv));

#ifndef DIM_DISCOVERY
    results_handler.vector_size = 43;
#endif

    // Setting up model
    auto mymodel = hd::MAP(results_handler.vector_size);
    if (results_handler.host) mymodel.setQueue(cl::sycl::queue{cl::sycl::host_selector{}});
    results_handler.autoset(mymodel);

    // Reading datasets, automaticly shuffles and splits
    auto data = hdna::readSpliceDataset(
        PROJECT_PATH_CMAKE "/examples/hdna/datasets/splice.data"
    );

    // Encoding atomic Hypervectors
    std::vector<char> dna_bases = 
        {'A', 'C', 'G', 'T'};

    auto plabels = hd::_processReduceLabels(data.first.labels);
    hd::lse::l_classifier c(
        plabels.unique_labels,
        mymodel.getVectorSize(),
        mymodel.getQueue()
    );
    
    results_handler.startGen();
    auto posMat = mymodel.genUnlabeledAtomicRep(SEQ_LENGTH);
    auto DNAMat = mymodel.genAtomicRep(dna_bases);
    results_handler.stopGen();

    // Training
    results_handler.startTraining();
    auto train_e = posMat
        .baseLevel(DNAMat,data.first.genes_data);

#ifdef DIM_DISCOVERY

    try {
        c.train_weights(train_e.hv_buffer, plabels.label_nidx_corr);
    } catch (std::exception &e) {
        return 1;
    }
    auto fit = c.test(train_e.hv_buffer, plabels.label_nidx_corr);
    std::cout << results_handler.vector_size << "," << fit * 100 << std::endl;

#else
    c.train_weights(train_e, plabels.label_nidx_corr);
    results_handler.stopTraining();

    auto fit = c.test(train_e, plabels.label_nidx_corr);
    std::cout << "Fit: " << fit * 100 << "%" << std::endl;

    // Testing
    results_handler.startTesting();
    auto test_e = posMat
        .baseLevel(DNAMat,data.second.genes_data);
    results_handler.success_rate = 
        c.test(test_e, data.second.labels)*100;
    results_handler.stopTesting();  

    // RESULTS
    results_handler.printToTerminal();
    results_handler.printToFile();
    return 0;
#endif
}