/**
 * @file readDataset.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt), Pedro André
 * @brief 
 * @version 0.1
 * @date 2024-01-21
 * 
 * @copyright GPL3
 * 
 */
#include "readDataset.hpp"
#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>

#define SPECIES_NAME_START 10
#define N_BASES_PER_GENE 657

hdna::dataset hdna::readBatsDataset(const std::string &full_path_to_file) {
    std::vector<std::string> labels = {};

    unsigned int start;
    unsigned int finish;

    std::string line_output;
    std::ifstream file(full_path_to_file);

    std::vector<std::string> genes_string;
    genes_string.emplace_back("");
    labels.emplace_back("");

    while (getline(file, line_output)) {
        start = line_output.find('|');
        finish = line_output.find('|', SPECIES_NAME_START);
        labels.back() = line_output.substr(start + 1, finish - start - 1);

        getline(file, line_output);
        genes_string.back() = line_output.substr(0, N_BASES_PER_GENE);

        genes_string.emplace_back("");
        labels.emplace_back("");
    }
    file.close();
    genes_string.pop_back();
    labels.pop_back();

    std::vector<std::vector<char>> genes_data(
        genes_string.size(),
        std::vector<char>(genes_string[0].size())
    );

    for (unsigned int i = 0; i < genes_string.size(); i++) {
        for (unsigned int j = 0; j < genes_string[i].size(); j++) {
            genes_data[i][j] = genes_string[i][j];
        }
    }

    return {genes_data,labels};
}


hdna::dataset readSpliceDataRaw(const std::string &filen){
    
    std::ifstream myfile; 
    myfile.open(filen);

    std::vector<std::vector<char>> genes;
    std::vector<std::string> labels;

    std::string class_str;
    std::string feature_str;

    if ( myfile.is_open() ) {     
        while ( myfile.good() ) {
            myfile >> class_str;
            class_str.pop_back();

            labels.push_back(class_str);
            
            myfile >> feature_str;
            myfile >> feature_str;

            std::vector<char> feature;
            for (unsigned int i = 0; i < feature_str.size(); i++) {
                feature.push_back(feature_str[i]);
            }

            genes.push_back(feature);
        }
    }
    return {genes,labels};
}

std::pair<hdna::dataset, hdna::dataset> hdna::readSpliceDataset(const std::string &full_path_to_file, bool shuffle, float ratio) {
    
    auto full_data = readSpliceDataRaw(full_path_to_file);
    
    std::vector<int> indexes(full_data.labels.size());
    std::iota(indexes.begin(), indexes.end(), 0);

    if (shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indexes.begin(), indexes.end(), g);
    }

    dataset train_data;
    dataset test_data;

    size_t train_size = indexes.size() * ratio;

    train_data.labels.reserve(train_size);
    train_data.genes_data.reserve(train_size);

    test_data.labels.reserve(indexes.size() - train_size);
    test_data.genes_data.reserve(indexes.size() - train_size);

    for (unsigned int i = 0; i < train_size; i++) {
        train_data.labels.push_back(full_data.labels[indexes[i]]);
        train_data.genes_data.push_back(full_data.genes_data[indexes[i]]);
    }

    for (unsigned int i = train_size; i < indexes.size(); i++) {
        test_data.labels.push_back(full_data.labels[indexes[i]]);
        test_data.genes_data.push_back(full_data.genes_data[indexes[i]]);
    }

    return {train_data,test_data};

}