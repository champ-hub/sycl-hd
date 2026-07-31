/**
 * @file mnist.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief MNIST example with baselevel encoder
 * @version 0.1
 * @date 2024-01-21
 * 
 * TODO: remake this example with same mnist reader as in mnist_rbf
 *
 * @copyright GPL3
 * 
 */
#define N_COLORS 256
#define IMG_SIZE 28*28

//#define SYCL_HD_DEBUG
//#define ACCURATE_TIME

#include <ResultsHandler.hpp>
#include <syclhd.hpp>
#include <readExeInputs.hpp>

#include "mnist/mnist_reader.hpp"


int main(int argc, char **argv) {

    // Gathering inputs
    ResultsHandler results_handler("MNIST",readExeInputs(argc, argv));

    auto mymodel = hd::MAP(results_handler.vector_size);
    if (results_handler.host) mymodel.setQueue(cl::sycl::queue{cl::sycl::host_selector{}});
    results_handler.autoset(mymodel);


    // Reading datasets
    mnist::MNIST_dataset<std::vector, std::vector<unsigned char>, unsigned char> dataset =
        mnist::read_dataset<std::vector, std::vector, unsigned char, unsigned char>(PROJECT_PATH_CMAKE "/examples/mnist/datasets");

    hd::index_2d_buffer train_images = hd::mat2RectBuff<unsigned char, unsigned short>(dataset.training_images);
    hd::index_2d_buffer test_images = hd::mat2RectBuff<unsigned char, unsigned short>(dataset.test_images);


    // Creating atomic representations
    results_handler.startGen();
    auto colors_rep = mymodel.genUnlabeledAtomicRep(N_COLORS, hd::full_level);
    auto pos_rep = mymodel.genUnlabeledAtomicRep(IMG_SIZE);
    results_handler.stopGen();


    // TRAINING
    results_handler.startTraining();
    auto associative_memory = pos_rep
        .baseLevel(colors_rep,train_images)
        .train(dataset.training_labels,results_handler.rt_steps,results_handler.alpha);
    results_handler.stopTraining();

    
    // TESTING
    results_handler.startTesting();

    auto encoded_entries = pos_rep.baseLevel(colors_rep,test_images);

    results_handler.success_rate =
        associative_memory.test(encoded_entries, dataset.test_labels)*100;
    
    results_handler.stopTesting();

    // RESULTS 
    results_handler.printToTerminal();
    results_handler.printToFile();

    return 0;
}