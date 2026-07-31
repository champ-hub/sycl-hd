/**
 * @file tmap_query.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-01
 * 
 * @copyright GPL3
 * 
 */

#include "tmap.hpp"
#include "../../core/macros.hpp"

namespace hd {

    COSINE_DISTANCE_MATRIX(ternary_hv_buffer,{
        t4_char v1 = acc_encoded_test_entries[i][k];
        t4_char v2 = acc_model_entries[j][k];

        num += v1.t0*v2.t0 +
               v1.t1*v2.t1 +
               v1.t2*v2.t2 +
               v1.t3*v2.t3
        ;

        den1 += v1.t0 * v1.t0 +
                v1.t1 * v1.t1 +
                v1.t2 * v1.t2 +
                v1.t3 * v1.t3
        ;

        den2 += v2.t0*v2.t0 +
                v2.t1*v2.t1 +
                v2.t2*v2.t2 +
                v2.t3*v2.t3
        ;
    })

    template <>
    float_2d_buffer matrix<ternary_hv_buffer>::distanceMatrix(
        matrix<ternary_hv_buffer> &to_query) {
      return cosineDistanceMatrix(this->hv_buffer, to_query.hv_buffer, this->q);
    }

    /** 
     * @brief MAP ternary get vector size
     *
     * Multiplies by 4 since each element in the buffer contains
     * is 1 byte that is divided into 4 field that store a ternary value 
     * 
     * @tparam  
     * @return size_t 
     */
    template <>
    size_t matrix<ternary_hv_buffer>::getVectorSize(){
        return hv_buffer.get_range()[1]*4;
    }

    template<> 
    float_1d_buffer matrix<ternary_hv_buffer>::getVarianceVector(const bool normalized){

        const size_t vectors_n = this->hv_buffer.get_range()[0];
        const size_t dimensionality = this->hv_buffer.get_range()[1];

        float_1d_buffer buff_to_return(dimensionality*4);

        this->q.submit([&](cl::sycl::handler &h){
            cl::sycl::accessor accVectors(this->hv_buffer,h,cl::sycl::read_only);
            cl::sycl::accessor accRet(buff_to_return,h,cl::sycl::write_only,cl::sycl::no_init);
            
            h.parallel_for(dimensionality,
            [=](cl::sycl::id<1> dim) {
                for (int N = 0; N<4; N++){

                    float mean = 0.0f, var = 0.0f;

                    for (size_t i = 0; i < vectors_n;i++){
                        mean += accVectors[i][dim][N];
                    }

                    mean /= vectors_n;
                    
                    for (size_t i = 0; i< vectors_n; i++){
                        const float diff = accVectors[i][dim][N]-mean;
                        var += diff*diff;
                    }

                    accRet[dim*4 + N] = var;
                }
            });
        });
        
        if (normalized) 
            varianceNormalize(buff_to_return,vectors_n,this->q);

        return buff_to_return;
    };

    template class matrix<ternary_hv_buffer>;

}