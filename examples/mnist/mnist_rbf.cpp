/**
 * @file mnist.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief MNIST example NeuralHD style
 * @version 0.1
 * @date 2024-03-06
 * 
 * Uses https://github.com/wichtounet/mnist for dataset reading
 * 
 * @copyright GPL3
 * 
 */

#include <ResultsHandler.hpp>
#include <syclhd.hpp>
#include <readExeInputs.hpp>

#define ACCURATE_TIME

// These values set by this flag relate to the comparison with
// https://ieeexplore.ieee.org/abstract/document/10363602 
#define COMPARE 0

static const int vec_size = 2000;

// Encoding is done in RBF style ManiHD (cosine of dot product)
static const hd::rbf_method rbf_method = hd::cos_dot;

// neuralHD Regeneration with a 20% ratio and max 200 iterations
static const float regen_ratio = 0.2;

// uses retraining alpha = 0.037 with adapthd retraining with 5 iterations
static const hd::retrain rt_method = hd::adapthd;
static const float alpha = 0.037;

#if COMPARE
static const int rt_steps = 5;
static const int regen_iter = 200;

#else

static const int rt_steps = 0;
static const int regen_iter = 0;

#endif

#define SHOW_FIT false

#define HDC_ONEAPI_DATA_READER 0
#if HDC_ONEAPI_DATA_READER

#include "readDataset.hpp"

#else

#include "mnist/mnist_reader.hpp"

/**
 * This function normalizes the input images data.
 *
 * @param img_data the 2D buffer containing the images data
 * @param q the SYCL queue for processing
 */
void normalize(hd::float_2d_buffer &img_data, cl::sycl::queue &q){
    q.submit([&](cl::sycl::handler &h) {
        cl::sycl::accessor acc(img_data, h, cl::sycl::read_write);
        h.parallel_for(img_data.get_range()[0], [=](cl::sycl::id<1> img) {
            float norm = 0.0;
            for (int px = 0; px < (28*28); px++){
                norm += acc[img][px]*acc[img][px];
            }
            norm = sqrt(norm);
            for (int px = 0; px < (28*28); px++){
                acc[img][px] /= norm;
            }
        });
    });
}

#endif

hd::matrix<hd::float_2d_buffer> encode(hd::model<float> &model, hd::float_2d_buffer &img_data){
    auto encoded = model.RBFKernelTrick(img_data, 0.0f, 1.0f, hd::cos_dot);
    return encoded;
}

int main(int argc, char **argv) {

    // Gathering inputs
    ResultsHandler results_handler("MNIST RBF",readExeInputs(argc, argv));

    // Override with comparison values
    results_handler.regen_iterations = regen_iter;
    results_handler.regen_ratio = regen_ratio;
    results_handler.alpha = alpha;
    results_handler.rt_steps= rt_steps;
    results_handler.vector_size = vec_size;

#if HDC_ONEAPI_DATA_READER

    char *testFile = strdup(DATA_PATH "mnist_test.choir_dat");
	mnist_c::Data test = mnist_c::readData(testFile);

    char* trainFile = strdup(DATA_PATH "mnist_train.choir_dat");
	mnist_c::Data train = mnist_c::readData(trainFile);

    //std::cout << "Train data size: " << train.data.size() << "x"<< train.data[0].size() << std::endl;
    //std::cout << "Test data size: " << test.data.size() << "x"<< test.data[0].size() << std::endl;

    hd::float_2d_buffer train_images = hd::mat2RectBuff(train.data);
    hd::float_2d_buffer test_images = hd::mat2RectBuff(test.data);

#else

    mnist::MNIST_dataset<std::vector, std::vector<unsigned char>, unsigned char> dataset =
        mnist::read_dataset<std::vector, std::vector, unsigned char, unsigned char>(PROJECT_PATH_CMAKE "/examples/mnist/datasets");

    hd::float_2d_buffer train_images = hd::mat2RectBuff<unsigned char, float>(dataset.training_images);
    hd::float_2d_buffer test_images = hd::mat2RectBuff<unsigned char, float>(dataset.test_images);

    { // Normalization
        cl::sycl::queue q;
        normalize(train_images, q);
        normalize(test_images, q);
    }

    struct {
        std::vector<unsigned char> labels;
    } train = {dataset.training_labels};
    struct {
        std::vector<unsigned char> labels;
    } test = {dataset.test_labels};

#endif

    // Set up model
    auto mymodel= hd::MAP(results_handler.vector_size);
    
    if (results_handler.host) mymodel.setQueue(cl::sycl::queue{cl::sycl::host_selector{}});

    results_handler.autoset(mymodel);
    
    //VECTOR GEN IS TIMED IN TRAINING
    results_handler.startGen();
    results_handler.stopGen();

    // TRAINING
    results_handler.startTraining();
    
    // Regeneration
    for (int i = 0; i < results_handler.regen_iterations; i++){
        auto AM = encode(mymodel,train_images)
            .train(train.labels,results_handler.rt_steps, results_handler.alpha,rt_method,SHOW_FIT);
        mymodel.regenerate(AM,{},results_handler.regen_ratio);
    } 

    // Final AM
    auto AM = encode(mymodel,train_images)
        .train(train.labels,results_handler.rt_steps, results_handler.alpha, rt_method,SHOW_FIT);
    results_handler.stopTraining();
    

    // TESTING
    results_handler.startTesting();
    results_handler.success_rate = [&](){
        auto test_rep = encode(mymodel, test_images);
        return AM.test(test_rep, test.labels);
    }()*100;
    results_handler.stopTesting();

    // RESULTS
    results_handler.printToTerminal();

    results_handler.printToFile();

    return 0;
}