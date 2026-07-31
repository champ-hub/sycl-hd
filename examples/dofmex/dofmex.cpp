/**
 * @file dofmex.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief SYCL-HDC Dollar of Mexico example
 * @version 2.0
 * @date 2024-01-28
 *
 *
 * @copyright GPL3
 * 
 */

#include <syclhd.hpp>
#include <iostream>
#include <vector>

//#define FPGA_EMULATOR
#ifdef FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
static const auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

template<typename K, typename V>
void print_map(std::unordered_map<K, V> const &m)
{
    for (auto const &pair: m) {
        std::cout << "{" << pair.first << ": " << pair.second << "}\n";
    }
}


int main(int argc, char **argv) {

    // Dollar of mexico atomic symbols
    std::vector<std::string> keys = {"Name","Capital","Currency"};
    std::vector<std::string> values = {"USA","WDC","Dollar",
                                       "Mexico","Mex. City","Peso"};
    // Model set up
#ifdef FPGA_EMULATOR
    auto mymodel = hd::BSC(8*100,cl::sycl::queue{selector});
#else
    auto mymodel = hd::BSC(8*100);
#endif
    std::cout << "Running on: " << mymodel.getDeviceName() << std::endl;

    //Generate atom rep
    auto keys_rep = mymodel.genAtomicRep(keys,hd::random,hd::ignore); // size 3*100
    auto values_rep = mymodel.genAtomicRep(values);         // size 6*100

    // Encode full representation
    std::vector<std::vector<std::string>> data = {{"USA","WDC","Dollar"},
                                                  {"Mexico","Mex. City","Peso"}};
                                        
    auto countries_rep = keys_rep.baseLevel(values_rep,data); // size 2*100
    auto full_rep = countries_rep.bindDown();             // size 1*100

    //Place in AM
    auto AM = keys_rep.stack(values_rep);
    std::cout << std::endl;

    // Querry
    auto dollar = values_rep["Dollar"];             // size 1*100
    auto query = full_rep.unbind(dollar);            // size 1*100

    auto result = AM.query(query)[0]; // [0] converts vector with 1 element to string


    // Should output: Peso
    std::cout << "The 'Dollar of Mexico' is: " << result << std::endl << std::endl;


    auto sim_mat = AM.distanceMatrix(query);
    std::vector<std::string> q_l = {"query"};
    
    std::cout.precision(6);
    std::cout << "SIMILARITY MATRIX" << std::endl;
    hd::print_debug_2dbuffer(sim_mat, AM.labels, q_l,10);
    std::cout<<std::endl;

    return 0;
}