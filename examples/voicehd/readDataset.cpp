/**
 * @file readDataset.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief VoiceHD dataset class implementation
 * @version 0.1
 * @date 2024-02-12
 * 
 * @copyright GPL3
 * 
 */
#include "readDataset.hpp"
#include <fstream>
#include <sstream>

dataset readDataset(const std::string &full_path_to_file) {
    std::ifstream file(full_path_to_file);
    std::string line_read;
    std::istringstream stream_line_read;
    std::string value_string;
    std::vector<float> freq_values;

    std::vector<std::string> labels;
    std::vector<std::vector<uint>> data;

    while (std::getline(file, line_read)) {
        freq_values = {};

        stream_line_read = std::istringstream(line_read);
        while (std::getline(stream_line_read, value_string, ' ')) {
            freq_values.push_back(stof(value_string));
        }

        labels.emplace_back(1, static_cast<char>(64 + freq_values.back()));
        freq_values.pop_back();

        data.emplace_back();
        for (const float &i: freq_values) {
            int aux = static_cast<int>((i + 1) * 10);
            if (aux == 20) {
                aux = 19;
            }
            data.back().push_back(aux);
        }
    }

    file.close();

    return dataset(hd::mat2RectIdxBuff(data),labels);
}