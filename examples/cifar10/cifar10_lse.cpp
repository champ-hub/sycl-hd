/**
 * @file cifar10_lse.cpp
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

#define N_U_CHARS 256
#define N_CLASSES 10
#define N_POS 3072

//#define DIM_DISCOVERY

/*
Dims, Fit

222,37.086
223,37.012
224,36.938
225,37.006
226,36.866
227,36.496
228,37.202
229,36.602
230,36.96
231,36.762
232,36.642
233,36.378
234,37.112
236,36.766
...
449,40.41
450,40.424
451,40.71
452,40.43
453,33.238
454,40.346
455,40.154
456,40.226
457,40.328
460,40.47
461,40.198
462,40.64
463,32.226
464,40.234
...
3074,50.318
*/


int main(int argc, char *argv[]) {

    ResultsHandler results_handler("CIFAR-10 VSLSC",readExeInputs(argc, argv));

#ifndef DIM_DISCOVERY // Override vector size from results handler
    results_handler.vector_size = N_U_CHARS;
#endif
    
    auto m = hd::MAP(results_handler.vector_size);
    if (results_handler.host) m.setQueue(cl::sycl::queue{cl::sycl::host_selector{}});
    results_handler.autoset(m);

    hd::lse::classifier c(m.getVectorSize(), N_CLASSES, m.getQueue());

    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    hd::index_2d_buffer train_i = hd::mat2RectIdxBuff(dataset.training_images);
    hd::index_1d_buffer train_l = hd::index_1d_buffer(dataset.training_labels.begin(), dataset.training_labels.end());
    hd::index_2d_buffer test_i = hd::mat2RectIdxBuff<>(dataset.test_images);
    hd::index_1d_buffer test_l = hd::index_1d_buffer(dataset.test_labels.begin(), dataset.test_labels.end());

    // Generating atomic Hypervectors
    results_handler.startGen();
    auto intensity_representation = m.genUnlabeledAtomicRep(N_U_CHARS,hd::full_level);
    auto position_color_vectors = m.genUnlabeledAtomicRep(N_POS);
    results_handler.stopGen();

    results_handler.startTraining();
    // Encoding entries (pseudo hidden layer)
    auto enc_train_entries = position_color_vectors.baseLevel(
            intensity_representation,
            train_i);

#ifdef DIM_DISCOVERY
    try {
        c.train_weights(enc_train_entries.hv_buffer, train_l);
    } catch (std::exception &e) {
        //std::cout << e.what() << std::endl;
        return 1;
    }

    results_handler.stopTraining();

    auto fit = c.test(enc_train_entries.hv_buffer,train_l)*100;
    std::cout << results_handler.vector_size << ",";
    std::cout  << fit << std::endl;

#else

    c.train_weights(enc_train_entries.hv_buffer, train_l);
    results_handler.stopTraining();
    
    auto fit = c.test(enc_train_entries.hv_buffer,train_l)*100;
    std::cout << "Fit: " << fit << "%" << std::endl;

    results_handler.startTesting();
    auto enc_test_entries = position_color_vectors.baseLevel(
        intensity_representation,
        test_i
    );
   
    results_handler.success_rate =
         c.test(enc_test_entries.hv_buffer,test_l)*100;
    results_handler.stopTesting();

    results_handler.printToFile();
    results_handler.printToTerminal();

#endif

    return 0;
}