/**
 * @file map_query.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-13
 * 
 * @copyright GPL3
 * 
 */

#include "map.hpp"

namespace hd {

    template <>
    float_2d_buffer matrix<float_2d_buffer>::distanceMatrix(
        matrix<float_2d_buffer> &to_query) {
      return cosineDistanceMatrix(this->hv_buffer, to_query.hv_buffer, this->q);
    }

    /**
     * @brief Generic get hyper-vector size
     * 
     * @tparam buffer_type 
     * @return size_t 
     */
    template <>
    size_t matrix<float_2d_buffer>::getVectorSize() {
      return hv_buffer.get_range()[1];
    }

    template class matrix<float_2d_buffer>;

}

namespace hd {
    template<> 
    float_1d_buffer matrix<float_2d_buffer>::getVarianceVector(const bool normalized){
        const size_t dimensionality = this->hv_buffer.get_range()[1];
        const size_t vectors_n = this->hv_buffer.get_range()[0];

        float_1d_buffer buff_to_return(dimensionality);

        this->q.submit([&](cl::sycl::handler &h){
            cl::sycl::accessor accVectors(this->hv_buffer,h,cl::sycl::read_only);
            cl::sycl::accessor accRet(buff_to_return,h,cl::sycl::write_only,cl::sycl::no_init);
            h.parallel_for(buff_to_return.get_range(),
            [=](cl::sycl::id<1> i) {
                float mean = 0.0f;
                for (int j = 0; j < vectors_n;j++){
                    mean += accVectors[j][i];
                }
                mean /= vectors_n;
                
                float var = 0.0f;
                for (int j= 0; j < vectors_n;j++){
                    float diff = accVectors[j][i]-mean;
                    var += diff*diff;
                }
                
                accRet[i] = var;
            });
        });
        
        if (normalized) 
            varianceNormalize(buff_to_return,vectors_n,this->q);

        return buff_to_return;
    };

}
