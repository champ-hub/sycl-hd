#include "readDataset.hpp"
#include <fstream>
//#include <algorithm>

std::vector<unsigned char> readFile(const std::string fn){
    std::ifstream testFile(fn, std::ios::binary);
    std::vector<unsigned char> fileContents((std::istreambuf_iterator<char>(testFile)),
                               std::istreambuf_iterator<char>());
    return fileContents;
}

void processRawFile(std::vector<unsigned char> &data, const bool remove_n_line){

    auto condition = [=](unsigned char &c){
        bool low_a = c < 'a';
        bool high_z = c > 'z';
        bool not_n_line = c != '\n';
        if ((low_a || high_z) && (not_n_line || remove_n_line))
            c = 'z' - 'a' + 1;
        else 
            c -= 'a';
    };

    for (auto &c : data)
        condition(c);

}

std::vector<std::vector<unsigned char>> splitNewLines(std::vector<unsigned char> &fileContents){
    std::vector<size_t> new_line_idx = {0};

    auto condition = [](unsigned char &c){
        return c == (unsigned char)('\n' - 'a');
    };

    for (int i = 0; i < fileContents.size(); i++){
        if (condition(fileContents[i]))
            new_line_idx.push_back(i+1);
    }

    std::vector<std::vector<unsigned char>> to_ret(new_line_idx.size()-1);

    for (size_t i = 1; i < new_line_idx.size(); i++)
        to_ret[i-1] = {fileContents.begin() + new_line_idx[i-1] , fileContents.begin() + new_line_idx[i] - 1 };

    return to_ret;
}