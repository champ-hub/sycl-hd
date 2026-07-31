/**
 * @file readDataset.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt), Pedro André
 * @brief 
 * @version 0.1
 * @date 2024-01-21
 * 
 * @copyright GPL3
 * 
 */
#ifndef DPHDC_READDATASET_HPP
#define DPHDC_READDATASET_HPP

#include <vector>
#include <string>

namespace hdna {

    struct dataset {
        std::vector<std::vector<char>> genes_data;
        std::vector<std::string> labels;
    };

dataset readBatsDataset(const std::string &full_path_to_file);

std::pair<dataset, dataset> readSpliceDataset(const std::string &full_path_to_file, bool shuffle = true, float ratio = 0.85);

}


#endif //DPHDC_READDATASET_HPP
