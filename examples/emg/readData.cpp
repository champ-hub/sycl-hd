/**
 * @file readData.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief  Read emg dataset functions adapted from HDCC
 * @version 0.1
 * @date 2024-05-16
 * 
 * @copyright GPL3
 * 
 */

#include "readData.hpp"
#include <fstream>

#define DATA_PATH PROJECT_PATH_CMAKE "/examples/emg/data/"


emg::data_t readDataPoints(const std::string &full_path_to_file) {
    std::ifstream file{full_path_to_file};
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file " + full_path_to_file);
    }
    std::vector<std::vector<int>> data;
    std::string line_read;
    std::istringstream stream_line_read;
    std::string value_string;
    while (std::getline(file, line_read)) {
        stream_line_read = std::istringstream(line_read);
        std::vector<int> line_values;
        while (std::getline(stream_line_read, value_string, ',')) {
            try {
                line_values.push_back(std::stoi(value_string));
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("Invalid argument exception while reading data from file " + full_path_to_file + ": " + std::string(e.what()));
            } catch (const std::out_of_range& e) {
                throw std::runtime_error("Out of range exception while reading data from file " + full_path_to_file + ": " + std::string(e.what()));
            }
        }
        data.push_back(line_values);
    }
    file.close();
    return hd::mat2RectIdxBuff(data);
}


emg::label_t readDataLabels(const std::string &full_path_to_file) {
    std::ifstream file{full_path_to_file};
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file " + full_path_to_file);
    }
    emg::label_t data{};
    std::string line_read;
    while (std::getline(file, line_read)) {
        try {
            data.push_back(std::stoi(line_read));
        } catch (const std::invalid_argument& e) {
            throw std::runtime_error("Invalid argument exception while reading labels from file " + full_path_to_file + ": " + std::string(e.what()));
        } catch (const std::out_of_range& e) {
            throw std::runtime_error("Out of range exception while reading labels from file " + full_path_to_file + ": " + std::string(e.what()));
        }
    }
    file.close();
    return data;
}




namespace emg{
    dataset readDataset(const std::string &full_path_to_patient){
        return {
            readDataPoints(full_path_to_patient + "_train_data"),
            readDataPoints(full_path_to_patient + "_test_data"),
            readDataLabels(full_path_to_patient + "_train_labels"),
            readDataLabels(full_path_to_patient + "_test_labels")
        };
    }

    dataset readDataset(const int patient){
        return readDataset(DATA_PATH "patient_" + std::to_string(patient));
    }
}

