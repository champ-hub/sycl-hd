#ifndef DPHDC_READDATASET_HPP
#define DPHDC_READDATASET_HPP

#include <vector>
#include <string>
#include <CL/sycl.hpp>

static const std::vector<std::string> language_names = {
    "Bulgarian", "Czech", "Danish", "German", "Greek", "English",
    "Estonian", "Finnish", "French", "Hungarian", "Italian", "Latvian",
    "Lithuanian", "Dutch", "Polish", "Portuguese", "Romanian", "Slovak",
    "Slovenian", "Spanish", "Swedish"};

static const std::array<std::string,21> file_names = {
    "bul.txt", "ces.txt", "dan.txt", "deu.txt", "ell.txt", "eng.txt", "est.txt",
    "fin.txt", "fra.txt", "hun.txt", "ita.txt", "lav.txt", "lit.txt", "nld.txt",
    "pol.txt", "por.txt", "ron.txt", "slk.txt", "slv.txt", "spa.txt", "swe.txt"};

static const uint n_languages = language_names.size();

static const uint n_chars = 28;

std::vector<unsigned char> readFile(const std::string fn);

void processRawFile(std::vector<unsigned char> &data, const bool remove_n_line = false);

std::vector<std::vector<unsigned char>> splitNewLines(std::vector<unsigned char> &fileContents);


#endif //DPHDC_READDATASET_HPP
