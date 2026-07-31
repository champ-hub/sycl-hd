/**
 * @file rbf_enc.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief RBF encoding helper methods
 * @version 0.1
 * @date 2024-02-13
 * 
 * @copyright GPL3
 * 
 */
#ifndef SYCLHD_RBF_HPP
#define SYCLHD_RBF_HPP

#include "definitions.hpp"

namespace hd {

    float_2d_buffer genBVectors(cl::sycl::range<2> Br, const float mean=0.0f, const float std_dev=1.0f, cl::sycl::queue *q=nullptr);

    float_2d_buffer maniHDRBF(float_2d_buffer &feature_vectors, float_2d_buffer &B_vectors, cl::sycl::queue &q);
    float_2d_buffer neuralHDRBF(float_2d_buffer &feature_vectors, float_2d_buffer &B_vectors, cl::sycl::queue &q);
    float_2d_buffer modneuralHDRBF(float_2d_buffer &feature_vectors, float_2d_buffer &B_vectors, cl::sycl::queue &q);

    float_2d_buffer neuralHDRBF(float_2d_buffer &feature_vectors, float_2d_buffer &B_vectors, float_2d_buffer &b_values, cl::sycl::queue &q);

}

#endif //SYCLHD_RBF_HPP