/**
 * @file cifar10.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-10-16
 * 
 * @copyright GPL3
 * 
 */

#include <syclhd.hpp>
#include "cifar10_reader.hpp"
#include "ResultsHandler.hpp"

#define N_LABELS 10
#define N_U_CHARS 256

int main(int argc, char **argv) {

    ResultsHandler results_handler("CIFAR-10",readExeInputs(argc, argv));
    
    auto m = hd::MAP(results_handler.vector_size);
    if (results_handler.host) m.setQueue(cl::sycl::queue{cl::sycl::host_selector{}});
    results_handler.autoset(m);

    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    auto train_i = hd::mat2RectIdxBuff(dataset.training_images);
    auto test_i = hd::mat2RectIdxBuff(dataset.test_images);


    // Generating atomic Hypervectors
    results_handler.startGen();
    auto intensity_representation = m.genUnlabeledAtomicRep(N_U_CHARS,hd::full_level);
    auto position_color_vectors = m.genUnlabeledAtomicRep(3072);
    results_handler.stopGen();

    results_handler.startTraining();
    auto associative_memory = position_color_vectors.baseLevel(
            intensity_representation,
            train_i)
            .train(
            dataset.training_labels,
            results_handler.rt_steps,
            results_handler.alpha,
            hd::onlinehd
    );
    results_handler.stopTraining();

    results_handler.startTesting();
    {
        auto encoded_test_entries = position_color_vectors.baseLevel(
            intensity_representation,
            test_i);
        
        results_handler.success_rate = 
            associative_memory.test(encoded_test_entries,dataset.test_labels)*100;
    }
    results_handler.stopTesting();

    results_handler.printToFile();
    results_handler.printToTerminal();

    return 0;
}