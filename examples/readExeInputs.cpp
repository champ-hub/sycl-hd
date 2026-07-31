#include "readExeInputs.hpp"
#include <cstring>
#include <string>
#include <stdexcept>

Inputs readExeInputs(int argc, char **argv) {
    Inputs inputs_read;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-vs") == 0) {
            if (argc > i + 1) {
                i++;
                inputs_read.vector_size = std::stoi(argv[i]);
            }
        }
        if (std::strcmp(argv[i], "-ng") == 0) {
            if (argc > i + 1) {
                i++;
                inputs_read.n_gram = std::stoi(argv[i]);
            }
        }
        if (std::strcmp(argv[i], "-rt") == 0) {
            if (argc > i + 1) {
                i++;
                inputs_read.rt_steps = std::stoi(argv[i]);
            }
        }
        if (std::strcmp(argv[i], "-a") == 0) {
            if (argc > i + 1) {
                i++;
                inputs_read.alpha = std::stof(argv[i]);
            }
        }
        if (std::strcmp(argv[i], "-rg") == 0) {
            if (argc > i + 1) {
                i++;
                inputs_read.regen_iterations = std::stoi(argv[i]);
            }
        }
        if (std::strcmp(argv[i], "-rgr") == 0) {
            if (argc > i + 1) {
                i++;
                inputs_read.regen_ratio = std::stof(argv[i]);
            }
        }
        if (std::strcmp(argv[i], "-host") == 0) {
            if (argc > i + 1) {
                i++;
                inputs_read.host = std::stoi(argv[i]);
            }
        }
    }

    if (inputs_read.vector_size <= 0) {
        throw std::invalid_argument("Vector size cannot be smaller than or equal to 0");
    }
    if (inputs_read.n_gram <= 0) {
        throw std::invalid_argument("N gram size cannot be smaller than or equal to 0");
    }
    if (inputs_read.rt_steps < 0) {
        throw std::invalid_argument("Retraining steps cannot be smaller than 0");
    }

    return inputs_read;
}