/**
 * @file query.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief Query helper methods
 * @version 0.1
 * @date 2024-02-01
 * 
 * @copyright GPL3
 * 
 */

#ifndef SYCLHD_QUERY_HPP
#define SYCLHD_QUERY_HPP

#include "definitions.hpp"


namespace hd {  // GET CLOSEST VEC AND SIM

    struct idx_sim_pair{
      index_1d_buffer indexes;
      float_1d_buffer similarities;  
    };

    idx_sim_pair getIdxSimVec(float_2d_buffer &sim_mat, cl::sycl::queue &q);
    index_1d_buffer idxOfHighestValueInMatRow(float_2d_buffer &sim_mat, cl::sycl::queue &q);
    float_2d_buffer cosineDistanceMatrix(float_2d_buffer &buff_keys, float_2d_buffer &buff_test, cl::sycl::queue &q);

    void varianceNormalize(float_1d_buffer &unnormVar, const int n_vecs, cl::sycl::queue &q);
}



#endif //SYCLHD_QUERY_HPP