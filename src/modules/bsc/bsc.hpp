/**
 * @file bsc.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-13
 * 
 * @copyright GPL3
 * 
 */

#ifndef HDC_BSC_HPP
#define HDC_BSC_HPP

#include "b8_char.hpp"
#include "../../core/model.hpp"

namespace hd {

    using binary_hv_buffer = cl::sycl::buffer<hd::b8_char, 2>;

    /**
     * @brief Create a Binary Spatter Codes Model
     * 
     * @param vs Vector Size / Dimensionality
     * @param queue SYCL Queue to Associate with
     * @param w Warnings
     * @return hd::model<b8_char> 
     */
    inline hd::model<b8_char> BSC(size_t vs = 10000, cl::sycl::queue queue = cl::sycl::queue{}, warning w = show){
        return hd::model<b8_char>(vs, "Binary Spatter Codes", 8,queue,w);
    }

    inline void print_debug_2dbuffer(binary_hv_buffer &buffer){
        auto range = buffer.get_range();
        cl::sycl::host_accessor acc_buff(buffer, cl::sycl::read_only);
        for (int i = 0; i < range[0]; i++){
            for (int j = 0; j < range[1]; j++)
                for(int N = 7; N >= 0; N--)
                    std::cout << int((acc_buff[i][j].ch >> N) & 1);
        std::cout << std::endl;
        }
    }   

}




#endif //HDC_BSC_HPP