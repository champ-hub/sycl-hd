/**
 * @file bsc_encoders.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-07
 * 
 * @copyright GPL3
 * 
 */

#include "bsc.hpp"
#include "../../core/macros.hpp"



namespace hd {
    BASIC_ENCODER(bind, binary_hv_buffer, {
        b8_char v = acc_this[acc_data[0]][dim];
        for (int i = 1; i < e_size; i++)
            v = v * acc_this[acc_data[i]][dim];
        acc_ret[e_i][dim] = v;
    })

    BASIC_ENCODER(bundle, binary_hv_buffer, {
        b8_char v = acc_this[acc_data[0]][dim]; 
        b8_ints acc(v);

        for (int i = 1; i < e_size; i++){
            v = acc_this[acc_data[i]][dim];
            acc += v;
        }

        v = acc > 0;  
        acc_ret[e_i][dim] = v;
    })

    BASIC_ENCODER_OP_SHORT(bundle, binary_hv_buffer,b8_ints,+=,v > 0)
    BASIC_ENCODER_OP_SHORT(bind, binary_hv_buffer,b8_char,*=,v)

    BIND_WITH(bindWith, binary_hv_buffer, *)

    template <>
    matrix<binary_hv_buffer> matrix<binary_hv_buffer>::unbind(matrix<binary_hv_buffer> &matrix_to_bind){
        return this->bindWith(matrix_to_bind);
    }

    /**
     * @brief Error: Attempted to invert a binary matrix
     * 
     * @tparam  
     * @return matrix<binary_hv_buffer> 
     */
    template <>
    matrix<binary_hv_buffer> matrix<binary_hv_buffer>::invert(){
        throw std::runtime_error("Attempted to invert a binary matrix");
    }

}

namespace hd { //Base Level

    BASELEVEL_ENCODER_NOT_SQUARE(binary_hv_buffer, {
        size_t j = acc_data[0];
        b8_char v = acc_levels[j][dim] * acc_bases[0][dim];
        b8_ints acc;
        acc = (v*2) - 1;

        for (int i = 0; i < acc_data.get_range()[0]; i++ ){
            size_t j = acc_data[i];
            v = acc_levels[j][dim] * acc_bases[i][dim];
            acc += (v*2) - 1;

        }

        v = acc > 0;
        acc_rets[0][dim] = v;

    })

    BASELEVEL_ENCODER_INTERNAL_FOR(binary_hv_buffer,{
        size_t i = lr[0];      // dt_p
        size_t i_ = lr[1];     // vs

        size_t k = acc_data[i][0];
        b8_char v = acc_levels[k][i_] * acc_bases[0][i_];
        
        b8_ints acc;
        acc = (v*2) - 1;

        for (unsigned int j = 1; j < n_base; j++){
            k = acc_data[i][j];
            v = acc_levels[k][i_] * acc_bases[j][i_];
            acc += (v*2) - 1;

        };

        v = acc > 0;
        acc_ret[i][i_] = v;

    })

}

namespace hd { // NGRAM


    NGRAM_BIND_BUNDLE(binary_hv_buffer, b8_ints, b8_char c(0) ,v > 0)

    NGRAM_BL(binary_hv_buffer, b8_ints, v > 0)

    template class matrix<binary_hv_buffer>;

}
