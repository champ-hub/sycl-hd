/**
 * @file voicehd.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-12
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

//#define FPGA_EMULATOR
#ifdef FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
static const auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

int main(int argc, char **argv) {

    ResultsHandler results_handler("VoiceHD", readExeInputs(argc, argv));

    auto mymodel = hd::MAP(results_handler.vector_size);

#ifdef FPGA_EMULATOR
    mymodel.setQueue(cl::sycl::queue{selector});
#endif

    if (results_handler.host) mymodel.setQueue(cl::sycl::queue{cl::sycl::host_selector{}});

    results_handler.autoset(mymodel);

    dataset train_dataset = readDataset(PROJECT_PATH_CMAKE "/examples/voicehd/dataset/isolet1+2+3+4.data");
    
    results_handler.startGen();
    auto representation = mymodel.genUnlabeledAtomicRep(REPRESENTATION_VECTOR_SIZE,hd::full_level);
    auto position_vectors = mymodel.genUnlabeledAtomicRep(N_POSITIONS);
    results_handler.stopGen();

    results_handler.startTraining();
    auto associative_memory = 
        position_vectors
        .baseLevel(representation,train_dataset.data)
        .train(train_dataset.labels,results_handler.rt_steps, 0,hd::voicehd,SHOW_FIT);

    results_handler.stopTraining();

    {
        dataset test_dataset = readDataset(PROJECT_PATH_CMAKE "/examples/voicehd/dataset/isolet5.data");
        results_handler.startTesting();
        auto encoded_test_entries = position_vectors.baseLevel(representation, test_dataset.data);
        results_handler.success_rate = associative_memory.test(encoded_test_entries, test_dataset.labels) * 100;
        results_handler.stopTesting();
    }

    results_handler.printToTerminal();
    results_handler.printToFile();

    return 0;
}