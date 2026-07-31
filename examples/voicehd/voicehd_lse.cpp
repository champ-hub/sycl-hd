/**
 * @file voicehd_hdcnn.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief VoiceHD using the HD-LSE classifier
 * @version 0.1
 * @date 2024-10-16
 * 
 * @copyright GPL3
 * 
 */


#include <syclhd.hpp>
#include <readExeInputs.hpp>
#include <ResultsHandler.hpp>
#include "readDataset.hpp"

#define REPRESENTATION_VECTOR_SIZE 20
#define N_POSITIONS 617

#define SHOW_FIT false

int main(int argc, char **argv) {

    ResultsHandler results_handler("VoiceHD VSLSC", readExeInputs(argc, argv));

    //=== OVERRIDE DIM SIZE ===
    results_handler.vector_size = 306;
    //===

    auto mymodel = hd::MAP(results_handler.vector_size);

    if (results_handler.host) mymodel.setQueue(cl::sycl::queue{cl::sycl::host_selector{}});

    results_handler.autoset(mymodel);

    dataset train_dataset = readDataset(PROJECT_PATH_CMAKE "/examples/voicehd/dataset/isolet1+2+3+4.data");
    
    results_handler.startGen();
    auto representation = mymodel.genUnlabeledAtomicRep(REPRESENTATION_VECTOR_SIZE,hd::full_level);
    auto position_vectors = mymodel.genUnlabeledAtomicRep(N_POSITIONS);
    results_handler.stopGen();

    // Processed training labels
    auto p_train_l = hd::_processReduceLabels(train_dataset.labels);
    
    // Create classifier
    hd::lse::l_classifier myclassifier(
        p_train_l.unique_labels,
        mymodel.getVectorSize(),
        mymodel.getQueue()
    );

    results_handler.startTraining();
    auto train_e = 
        position_vectors
        .baseLevel(representation,train_dataset.data);
    
    myclassifier.train_weights(train_e.hv_buffer, p_train_l.label_nidx_corr);
    results_handler.stopTraining();
    
    { // GET FIT
        auto fit = 
            myclassifier.test(train_e.hv_buffer,p_train_l.label_nidx_corr)*100;
        std::cout << "Train fit: " << fit << "%" << std::endl;
    }

    dataset test_dataset = readDataset(PROJECT_PATH_CMAKE "/examples/voicehd/dataset/isolet5.data");
    
    results_handler.startTesting();
    auto encoded_test_entries = position_vectors.baseLevel(representation, test_dataset.data);
    results_handler.success_rate = 
        myclassifier.test(encoded_test_entries.hv_buffer, test_dataset.labels)*100;
    results_handler.stopTesting();


    results_handler.printToTerminal();
    results_handler.printToFile();

    return 0;
}