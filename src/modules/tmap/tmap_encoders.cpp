/**
 * @file tmap_encoders.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-07
 * 
 * @copyright GPL3
 * 
 */

#include "tmap.hpp"
#include "../../core/macros.hpp"

namespace hd {
    BASIC_ENCODER(bind, ternary_hv_buffer, {
        t4_char v = acc_this[acc_data[0]][dim];
        for (int i = 1; i < e_size; i++) {
            t4_char thisv = acc_this[acc_data[i]][dim];
            v *= thisv;
        }
        acc_ret[e_i][dim] = v;
    })

    BASIC_ENCODER(bundle, ternary_hv_buffer, {
        t4_char v = acc_this[acc_data[0]][dim]; 
        t4_ints accum = v;

        for (int i = 1; i < e_size; i++){
            v = acc_this[acc_data[i]][dim];
            accum += v;
        }
        
        // NORMALIZATION OPERATOR SEE t4_char.hpp
        v = accum.normalize();

        acc_ret[e_i][dim] = v;
    })

    BASIC_ENCODER_OP_SHORT(bundle, ternary_hv_buffer, t4_ints,+=, v.normalize())
    BASIC_ENCODER_OP_SHORT(bind, ternary_hv_buffer, t4_char,*=, v)

    BIND_WITH(bindWith, ternary_hv_buffer, *)

    template <>
    matrix<ternary_hv_buffer> matrix<ternary_hv_buffer>::unbind(matrix<ternary_hv_buffer> &matrix_to_bind){
        return this->bindWith(matrix_to_bind);
    }

    // This just throws an error because
    // the ternary matrix is its own inverse, might change this behaviour in the future
    template <>
    matrix<ternary_hv_buffer> matrix<ternary_hv_buffer>::invert(){
        throw std::runtime_error("Attempted to invert a ternary matrix");
    }
    
}

namespace hd { // Base Level
    BASELEVEL_ENCODER_INTERNAL_FOR(ternary_hv_buffer,{
        size_t i = lr[0];      // dt_p
        size_t i_ = lr[1];     // vs

        size_t k = acc_data[i][0];
        t4_char v;
        t4_char vl;
        t4_char vb;
        vl.ch = acc_levels[k][i_].ch;
        vb.ch = acc_bases[0][i_].ch;

        t4_ints accum = vl*vb;

        for (unsigned int j = 1; j < n_base; j++){
            size_t k = acc_data[i][j];
            vl.ch = acc_levels[k][i_].ch;
            vb.ch = acc_bases[j][i_].ch;
            accum += vl*vb;
        };

        // NORMALIZATION OPERATOR SEE t4_char.hpp
        v = accum.normalize();
        acc_ret[i][i_].ch = v.ch;

    })
}


namespace hd { 

    NGRAM_BIND_BUNDLE(ternary_hv_buffer, t4_ints, t4_char c(1) ,v.normalize())

    NGRAM_BL(ternary_hv_buffer, t4_ints, v.normalize())

    template class matrix<ternary_hv_buffer>;
} 