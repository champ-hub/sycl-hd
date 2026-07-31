/**
 * @file readDataset.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief VoiceHD dataset class header
 * @version 0.1
 * @date 2024-02-12
 * 
 * @copyright GPL3
 * 
 */
#ifndef DPHDC_READDATASET_HPP
#define DPHDC_READDATASET_HPP

#include <vector>
#include <string>
#include <syclhd.hpp>

class dataset {
    public:
    hd::index_2d_buffer data;
    const std::vector<std::string> labels;
    dataset(hd::index_2d_buffer &&data_set, std::vector<std::string> const& lbs):
    data(data_set), labels(lbs){}
};

dataset readDataset(const std::string &full_path_to_file);

#endif //DPHDC_READDATASET_HPP
