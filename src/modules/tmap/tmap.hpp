/**
 * @file tmap.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-13
 * 
 * @copyright GPL3
 * 
 */

#ifndef HDC_TMAP_HPP
#define HDC_TMAP_HPP

#define HDC_TMAP_NAME "Ternary MAP"

#include "../../core/model.hpp"
#include "t4_char.hpp"

namespace hd {

    using ternary_hv_buffer = cl::sycl::buffer<t4_char, 2>;

    /**
     * @brief Create a Ternary Multiply Add Permute model
     * 
     * @param vs Vector Size / Dimensionality
     * @param queue SYCL Queue
     * @param w Warnings
     * @return hd::model<t4_char> 
     */
    inline hd::model<t4_char> TMAP(size_t vs = 10000, cl::sycl::queue queue = cl::sycl::queue{}, warning w = show){
        return hd::model<t4_char>(vs,HDC_TMAP_NAME,4,queue,w);
    }

    inline void print_debug_2dbuffer(cl::sycl::buffer<t4_char,2> &buffer){
        auto range = buffer.get_range();
        cl::sycl::host_accessor acc_buff(buffer, cl::sycl::read_only);
        for (int i = 0; i < range[0]; i++){
            for (int j = 0; j < range[1]; j++){
                std::cout << std::right <<std::setw(3)
                          << acc_buff[i][j].t3; 
                std::cout << std::right <<std::setw(3)
                          << acc_buff[i][j].t2;
                std::cout << std::right <<std::setw(3)
                          << acc_buff[i][j].t1; 
                std::cout << std::right <<std::setw(3)
                          << acc_buff[i][j].t0;}
        std::cout << std::endl;
        }
    }

}

#endif //HDC_TMAP_HPP