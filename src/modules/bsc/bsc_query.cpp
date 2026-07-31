/**
 * @file query.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-01
 * 
 * @copyright GPL3
 * 
 */


#include "bsc.hpp"

namespace hd { // NORMALIZED HAMMING DISTANCE

    float_2d_buffer nhammingDistanceMatrix(binary_hv_buffer &buff_keys, binary_hv_buffer &buff_test, cl::sycl::queue &q) {

        cl::sycl::range<2> v_range(
            buff_test.get_range()[0],
            buff_keys.get_range()[0]
        );

        float_2d_buffer buff_distance_vectors(v_range);

        size_t vs = buff_keys.get_range()[1];

        q.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc_dist_vectors(buff_distance_vectors, h, cl::sycl::write_only, cl::sycl::no_init);
            cl::sycl::accessor acc_test_entries(buff_test, h,cl::sycl::read_only);
            cl::sycl::accessor acc_model_entries(buff_keys, h, cl::sycl::read_only);
            h.parallel_for(v_range, [=](cl::sycl::id<2> local_range) {
                size_t i = local_range[0];
                size_t j = local_range[1];
                float sim = 0;

                for (size_t k = 0; k < vs; k++){
                    unsigned char v = ~(acc_test_entries[i][k].ch ^ acc_model_entries[j][k].ch);
                    for(int N = 0; N < 8; N++)
                        sim += (((v >> N) & 1)/8.0);
                }

                acc_dist_vectors[local_range] = (sim/vs)*2-1;
            });
        });
        return buff_distance_vectors;
    }

    template<>
    float_2d_buffer matrix<binary_hv_buffer>::distanceMatrix(matrix<binary_hv_buffer> &to_query){
        return nhammingDistanceMatrix(this->hv_buffer, to_query.hv_buffer, this->q);
    }

    /**
     * @brief BSC get hypervector size
     *
     * Multiplies by 8 since each element in the buffer contains 8 bits
     * 
     * @tparam  
     * @return size_t 
     */
    template <>
    size_t matrix<binary_hv_buffer>::getVectorSize(){
        return hv_buffer.get_range()[1]*8;
    }

    /**
     * Calculate the variance vector for this matrix.
     *
     * @return float_1d_buffer The variance vector
     *
     */
    template<> 
    float_1d_buffer matrix<binary_hv_buffer>::getVarianceVector(const bool normalized){
        const size_t vectors_n = this->hv_buffer.get_range()[0];
        const size_t dimensionality = this->hv_buffer.get_range()[1];

        float_1d_buffer buff_to_return(dimensionality*8);

        this->q.submit([&](cl::sycl::handler &h){
            cl::sycl::accessor accVectors(this->hv_buffer,h,cl::sycl::read_only);
            cl::sycl::accessor accRet(buff_to_return,h,cl::sycl::write_only,cl::sycl::no_init);
            h.parallel_for(dimensionality,
            [=](cl::sycl::id<1> dim) {
                for (int N = 0; N<8; N++){

                    float mean = 0.0f,var = 0.0f;

                    for (size_t i = 0; i < vectors_n;i++){
                        mean += accVectors[i][dim][N];
                    }

                    mean /= vectors_n;
                    
                    for (size_t i = 0; i< vectors_n; i++){
                        const float diff = accVectors[i][dim][N]-mean;
                        var += diff*diff;
                    }

                    accRet[dim*8 + N] = var;
                }
            });
        });
        
        if (normalized) 
            varianceNormalize(buff_to_return,vectors_n,this->q);

        return buff_to_return;
    };

    template class matrix<binary_hv_buffer>;

}