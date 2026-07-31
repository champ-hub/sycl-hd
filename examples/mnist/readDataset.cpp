/**
 * @file readDataset.cpp
 * @author Ian
 * @brief  Read dataset taken from HDConeAPI
 * @version 0.1
 * @date 2024-05-25
 * 
 * @copyright GPL3
 * 
 */
 
#include "readDataset.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>

namespace mnist_c {

    fvec normalize(fvec input) {
        float inner = std::inner_product(input.begin(), input.end(), input.begin(), 0.0);
        float norm = std::sqrt(inner);
        for(int i = 0; i < input.size(); i++) {
            input[i] = input[i] / norm;
        }
        return input;
    }

    Data readData(char *filename) {
        std::ifstream testFile(filename, std::ifstream::binary);
        char *holder = (char *)malloc(4 * sizeof(char));
        testFile.read(holder, 4 * sizeof(char));
        int32_t numFeatures;
        memcpy(&numFeatures, &holder[0], sizeof(numFeatures));
        testFile.read(holder, 4 * sizeof(char));
        int32_t numClasses;
        memcpy(&numClasses, &holder[0], sizeof(numClasses));
        fmat testData;
        ivec testLabels;
        while(testFile.good()) {
            fvec vect(numFeatures, 0.0);
            bool zero = true;
            for(int i = 0; i < numFeatures; i++) {
                testFile.read(holder, 4 * sizeof(char));
                float val;
                memcpy(&val, &holder[0], sizeof(val));
                vect[i] = val;
                if(val != 0.0) {
                    zero = false;
                }
            } 
            testFile.read(holder, 4 * sizeof(char));
            int label;
            memcpy(&label, &holder[0], sizeof(label));
            if(!zero) {
                vect = normalize(vect);
                testData.push_back(vect);
                testLabels.push_back(label);
            }
        }
        free(holder);
        testFile.close();
        Data ret;
        ret.data = testData;
        ret.labels = testLabels;
        ret.numClasses = numClasses;
        ret.numFeatures = numFeatures;
        return ret;
    }

}