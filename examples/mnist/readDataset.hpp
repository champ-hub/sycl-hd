/**
 * @file readDataset.hpp
 * @author Ian
 * @brief Read dataset taken from HDConeAPI
 * @version 0.1
 * @date 2024-05-25
 * 
 * @copyright GPL3
 * 
 */

#ifndef READ_MNIST_CHOIRDAT
#define READ_MNIST_CHOIRDAT

#define DATA_PATH PROJECT_PATH_CMAKE "/examples/mnist/datasets/"

#include <cstdint>
#include <vector>

namespace mnist_c {

    typedef std::vector<std::vector<float>> fmat;
    typedef std::vector<float> fvec;
    typedef std::vector<int> ivec;

    typedef struct{
        fmat data;
        ivec labels;
        int32_t numFeatures;
        int32_t numClasses;
    } Data;

    fvec normalize(fvec input);
    Data readData(char *filename);

}

#endif //READ_MNIST_CHOIRDAT
