/**
 * @file language_lse.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief LSE Classifier for language example 
 * Poor accuracy
 * @version 0.1
 * @date 2024-10-17
 * 
 * @copyright GPL3
 * 
 */
#include "readDataset.hpp"
#include <syclhd.hpp>
#include <ResultsHandler.hpp>

//#define DIM_DISCOVERY

hd::matrix<hd::float_2d_buffer> encode(
    hd::matrix<hd::float_2d_buffer>& char_rep,
    std::vector<std::vector<unsigned char>> & data, 
    const size_t ngram
){
    return char_rep.ngram(data, ngram, hd::shift_left, true, true);
}

int main(int argc, char **argv) {

    ResultsHandler results_handler("Language VSLSC", readExeInputs(argc, argv));
    
#ifndef DIM_DISCOVERY // Override vector size from results handler
    results_handler.vector_size = 21;
#endif

    auto model = hd::MAP(results_handler.vector_size);

    if (results_handler.host) model.setQueue(cl::sycl::queue{cl::sycl::host_selector{}});

    results_handler.autoset(model);;
    
    std::vector<std::vector<unsigned char>> training_data(n_languages);
    for (size_t i = 0; i < n_languages; i++){
        training_data[i] = readFile(PROJECT_PATH_CMAKE "/examples/language/datasets/training/" + file_names[i]);
        processRawFile(training_data[i], true);
    }

    results_handler.startGen(); 
    auto char_rep = model.genUnlabeledAtomicRep(n_chars);
    results_handler.stopGen();

    //std::cout << "READ DATASET AND INITIALIZED REPRESENTATION " << std::endl;
    
    // Create classifier
    hd::lse::l_classifier myclassifier(
        language_names,
        model.getVectorSize(),
        model.getQueue()
    );

    results_handler.startTraining();
    auto train_e = 
        encode(char_rep, training_data, results_handler.ngram);
    

#ifdef DIM_DISCOVERY

    try{
        myclassifier.train_weights(train_e.hv_buffer, language_names);
        auto fit = myclassifier.test(train_e.hv_buffer, language_names);
        std::cout << results_handler.vector_size << "," << fit*100 << std::endl;
        return 0;
    }
    catch(std::exception &e){
        return 1;
    }


#else

    myclassifier.train_weights(train_e.hv_buffer, language_names);
    results_handler.stopTraining();

    //std::cout << "FINISHED TRAINING, READING TESTING" << std::endl;

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
        auto encoded_entries = encode(char_rep, testing_data[i], results_handler.ngram);
        auto this_result = myclassifier.classify(encoded_entries.hv_buffer);
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
#endif
}
