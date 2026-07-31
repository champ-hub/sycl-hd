/**
 * @file map.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-13
 * 
 * @copyright GPL3
 * 
 */


#include "../../core/model.hpp"

#ifndef HDC_MAP_HPP
#define HDC_MAP_HPP


namespace hd { // SPECIALIZED PROBLEM CHILDREN

    /**
     * @brief Create a Multiply Add Permute Model
     * 
     * @param vs Vector Size / Dimensionality
     * @param queue SYCL Queue
     * @param w Warnings
     * @return hd::model<float> 
     */
    inline hd::model<float> MAP(size_t vs = 10000, cl::sycl::queue queue = cl::sycl::queue{}, warning w = show){
        return hd::model<float>(vs,"Multiply-Add-Permute",1,queue,w);
    }
}

#endif //HDC_MAP_HPP

