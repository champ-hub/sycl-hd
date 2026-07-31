/**
 * @file mnist_lse.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-10-15
 * 
 * @copyright GPL3
 * 
 */
#include <mnist/mnist_reader.hpp>
#include <syclhd.hpp>
#include <ResultsHandler.hpp>

#define N_COLORS 256
#define IMG_SIZE 28*28
#define N_CLASSES 10

static const float train_ratio = 0; // >= 0 and < 1 where 0 is full
static const float test_ratio = 0; // >= 0 and < 1 where 0 is full

static const int training_limit = train_ratio*60000;
static const int test_limit = test_ratio*10000;

int main(int argc, char **argv) {

    ResultsHandler results_handler("MNIST VSLSC",readExeInputs(argc, argv));
    
    //===== OVERRIDE
    results_handler.vector_size = IMG_SIZE;
    //=====

    auto mymodel = hd::MAP(results_handler.vector_size);
    if (results_handler.host) mymodel.setQueue(cl::sycl::queue{cl::sycl::host_selector{}});
    results_handler.autoset(mymodel);

    hd::lse::classifier myclassifier(mymodel.getVectorSize(), N_CLASSES, mymodel.getQueue());

    mnist::MNIST_dataset dataset =
        mnist::read_dataset<std::vector, std::vector, unsigned short, unsigned char>(
            PROJECT_PATH_CMAKE "/examples/mnist/datasets",
            training_limit,
            test_limit
        );

    hd::index_2d_buffer train_images = hd::mat2RectBuff(dataset.training_images);
    hd::index_2d_buffer test_images = hd::mat2RectBuff(dataset.test_images);

    hd::index_1d_buffer train_labels(dataset.training_labels.begin(), dataset.training_labels.end());
    hd::index_1d_buffer test_labels(dataset.test_labels.begin(), dataset.test_labels.end());

    // Creating atomic representations
    results_handler.startGen();
    auto colors_rep = mymodel.genUnlabeledAtomicRep(N_COLORS, hd::full_level);
    auto pos_rep = mymodel.genUnlabeledAtomicRep(IMG_SIZE);
    results_handler.stopGen();

    // ENCODE TRAIN SAMPLES
    results_handler.startTraining();
    auto train_entries = 
        pos_rep
        .baseLevel(colors_rep,train_images);
    // TRAIN WEIGHTS
    myclassifier.train_weights(train_entries.hv_buffer, train_labels);
    results_handler.stopTraining();

    // GET FIT
    auto fit = myclassifier.test(train_entries.hv_buffer, train_labels);
    std::cout << "Train fit: " << fit*100 <<"%"<< std::endl;

    results_handler.startTesting();
    // ENCODE TEST SAMPLES
    auto test_entries = 
        pos_rep
        .baseLevel(colors_rep,test_images);

    // GET ACCURACY
    results_handler.success_rate = 
        myclassifier.test(test_entries.hv_buffer, test_labels) * 100;

    results_handler.stopTesting();

    results_handler.printToTerminal();
    results_handler.printToFile();

    return 0;
}