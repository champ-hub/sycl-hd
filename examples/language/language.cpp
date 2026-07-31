/**
 * @file language.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief EU Language Identification Example
 * @version 2.0
 * @date 2023-12-22
 * 
 * @copyright GPL3
 * 
 */
#include "readDataset.hpp"
#include <syclhd.hpp>
#include <ResultsHandler.hpp>


int main(int argc, char **argv) {

    ResultsHandler results_handler("Language", readExeInputs(argc, argv));

    std::cout << "Using n-grams of size " << results_handler.ngram << std::endl;

    auto model = hd::MAP(results_handler.vector_size);

    if (results_handler.host) model.setQueue(cl::sycl::queue{cl::sycl::host_selector{}});

    results_handler.autoset(model);
    
    std::vector<std::vector<unsigned char>> training_data(n_languages);
    for (size_t i = 0; i < n_languages; i++){
        training_data[i] = readFile(PROJECT_PATH_CMAKE "/examples/language/datasets/training/" + file_names[i]);
        processRawFile(training_data[i], true);
    }

    results_handler.startGen(); 
    auto char_rep = model.genUnlabeledAtomicRep(n_chars);
    results_handler.stopGen();

    std::cout << "READ DATASET AND INITIALIZED REPRESENTATION " << std::endl;

    results_handler.startTraining();
    auto associative_memory = char_rep
        .ngram(training_data, results_handler.ngram, hd::shift_left, true, true)
        .train(language_names, results_handler.rt_steps);
    results_handler.stopTraining();

    std::cout << "FINISHED TRAINING, READING TESTING" << std::endl;

    std::vector<std::vector<std::vector<unsigned char>>> testing_data(n_languages);
    for (size_t i = 0; i < n_languages; i++){
        auto raw_file_stream = readFile(PROJECT_PATH_CMAKE "/examples/language/datasets/testing/" + file_names[i]);
        processRawFile(raw_file_stream);
        testing_data[i] = splitNewLines(raw_file_stream);
    }

    std::cout << "FINISHED READING, STARTING TESTING" << std::endl;

    std::vector<cl::sycl::buffer<unsigned short, 1>> results;
    results.reserve(n_languages);

    results_handler.startTesting();

    for (size_t i = 0; i < n_languages; i++){
        auto encoded_entries = char_rep
            .ngram(testing_data[i], results_handler.ngram, hd::shift_left, true, true);
        auto this_result = associative_memory.queryIdx(encoded_entries);
        results.emplace_back(this_result);
    }
    results_handler.stopTesting();

    std::cout << "FINISHED TESTING, DISPLAYING INDIVIDUAL RESULTS" << std::endl;

    cl::sycl::buffer<double, 1> buff_accuracy(n_languages);
    for (size_t i = 0; i < 21; i++){
        model.getQueue().submit([&](cl::sycl::handler &h){
            cl::sycl::accessor acc_results(results[i], h, cl::sycl::read_only);
            cl::sycl::accessor acc_acc(buff_accuracy, h, cl::sycl::write_only, cl::sycl::no_init);
            size_t range = results[i].get_range()[0];
            h.single_task([=](){
                int success = 0;
                for (size_t j = 0; j < range; j++) {
                    success += (int)(acc_results[j] == i);
                }
                acc_acc[i] = (double) success / range * 100;
            });
        });
    }

    cl::sycl::host_accessor accuracy(buff_accuracy, cl::sycl::read_only);
    for (size_t i = 0; i < n_languages; i++){
        std::cout << "Accuracy for: " <<  language_names[i] << " ->  " << accuracy[i] << "%" << std::endl;
    }

    auto acc_sum = std::accumulate(accuracy.begin(), accuracy.end(), 0.0);
    results_handler.success_rate = acc_sum / accuracy.size();
    results_handler.printToTerminal();
    results_handler.printToFile();

    return 0;
}
