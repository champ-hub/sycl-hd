/**
 * @file map_encoders.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-13
 * 
 * @copyright GPL3
 * 
 */

#include "map.hpp"
#include "../../core/macros.hpp"


namespace hd { // BIND BUNDLE
    template<>
    matrix<float_2d_buffer> model<float>::RBFKernelTrick(float_2d_buffer &feature_vectors_buff, const float mean, const float std_dev, const rbf_method method){
        return _RBFKernelTrickFloatMat(feature_vectors_buff,mean,std_dev,method);
    }

    template<>
    matrix<float_2d_buffer> model<float>::RBFKernelTrick(std::vector<std::vector<float>> &feature_vectors_mat, const float mean, const float std_dev, const rbf_method method){
        return _RBFKernelTrickFloatMat(feature_vectors_mat,mean,std_dev,method);
    }

    BASIC_ENCODER(bind, float_2d_buffer, {
        float v = acc_this[acc_data[0]][dim];
        for (int i = 1; i < e_size; i++)
            v *= acc_this[acc_data[i]][dim];
        acc_ret[e_i][dim] = v;
    })

    BASIC_ENCODER(bundle, float_2d_buffer, {
        float v = acc_this[acc_data[0]][dim];
        for (int i = 1; i < e_size; i++)
            v += acc_this[acc_data[i]][dim];
        acc_ret[e_i][dim] = v;
    })

    BASIC_ENCODER_OP_SHORT(bundle, float_2d_buffer, float, +=, v)
    BASIC_ENCODER_OP_SHORT(bind, float_2d_buffer, float, *=, v)


    BIND_WITH(bindWith, float_2d_buffer, *)

    BIND_WITH(unbind, float_2d_buffer, /)

    /**
     * @brief Generic matrix invert operation
     *
     * For float type associated buffers
     * 
     * @tparam buffer_type 
     * @return matrix<buffer_type> 
     */
    template <>
    matrix<float_2d_buffer> matrix<float_2d_buffer>::invert() {
        matrix<float_2d_buffer> n_mat(this->hv_buffer.get_range(), this->q);
        this->q.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc_ret(n_mat.hv_buffer, h, cl::sycl::write_only,cl::sycl::no_init);
            cl::sycl::accessor acc_this(this->hv_buffer, h, cl::sycl::read_only);
            h.parallel_for(this->hv_buffer.get_range(),[=](cl::sycl::id<2> i) { 
                acc_ret[i] = 1 / acc_this[i]; 
                });
        });
        return n_mat;
    }

    /**
     * @brief Error: Attempted to invert an integer matrix
     * 
     * @tparam  
     * @return matrix<int_2d_buffer> 
     */
    template <>
    matrix<int_2d_buffer> matrix<int_2d_buffer>::invert(){
        throw std::runtime_error("Attempted to invert an integer matrix");
    }

    /**
     * @brief Error Attempted to unbind integer matrices
     * 
     * @tparam  
     * @param matrix_to_bind 
     * @return matrix<int_2d_buffer> 
     */
    template <>
    matrix<int_2d_buffer> matrix<int_2d_buffer>::unbind(matrix<int_2d_buffer> &matrix_to_bind){
        throw std::runtime_error("Attempted to unbind integer matrices");
    }

}

namespace hd { // Base Level

    BASELEVEL_ENCODER_NOT_SQUARE(float_2d_buffer,{
        for (int i = 0; i < acc_data.get_range()[0]; i++ ){
            size_t j = acc_data[i];
            float v = acc_levels[j][dim] * acc_bases[i][dim];
            acc_rets[0][dim] += v;
        }
    })

    BASELEVEL_ENCODER_INTERNAL_FOR(float_2d_buffer, {
        size_t i = lr[0];      // dt_p

        size_t i_ = lr[1];     // vs
        size_t k = acc_data[i][0]; // k should never be bigger than this_size 

        float v = acc_levels[k][i_] * acc_bases[0][i_];

        for (unsigned int j = 1; j < n_base; j++){
            k = acc_data[i][j];
            v += acc_levels[k][i_] * acc_bases[j][i_];
        }

        acc_ret[i][i_] = v;

    })
}

namespace hd { // NGRAM

    NGRAM_BIND_BUNDLE(float_2d_buffer,float,float c = 1,v)
    
    NGRAM_BL(float_2d_buffer, float, v)

    template class matrix<float_2d_buffer>;
}