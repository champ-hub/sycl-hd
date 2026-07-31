/**
 * @file readData.hpp
 * @author 
 * @brief Read emg dataset functions adapted from HDCC
 * @version 0.1
 * @date 2024-05-16
 * 
 * @copyright GPL3
 * 
 */

#pragma once
#include <syclhd.hpp>


namespace emg {

    using data_t = hd::index_2d_buffer;
    using label_t = std::vector<uint>;

    struct dataset{
        data_t train_data;
        data_t test_data;
        label_t train_labels;
        label_t test_labels;
    };

    dataset readDataset(const std::string &full_path_to_patient);
    dataset readDataset(const int patient);

}