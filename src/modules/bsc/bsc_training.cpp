/**
 * @file bsc_training.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-09
 * 
 * @copyright GPL3
 * 
 */

#include "bsc.hpp"

namespace hd { // Convert to and from float matrix

    template<>
    matrix<float_2d_buffer> matrix<binary_hv_buffer>::toFloatMatrix()
    {
        cl::sycl::range r(this->hv_buffer.get_range()[0],this->hv_buffer.get_range()[1]*8);
        matrix<float_2d_buffer> to_ret(r,this->q);

        q.submit([&](cl::sycl::handler &h){
            auto bin_acc = this->hv_buffer.get_access(h,cl::sycl::read_only);
            auto float_acc = to_ret.hv_buffer.get_access(h,cl::sycl::write_only, cl::sycl::no_init);
            h.parallel_for(r,[=](cl::sycl::id<2> lr){
                size_t i = lr[0];
                size_t j = lr[1];

                int j_ = j/8;
                int N = j%8;

                char v = bin_acc[i][j_].ch;
                float_acc[i][j] = ((v >> N) & 0b00000001)*2 - 1;
            });
        });

        return to_ret;
    }

    template<>
    void matrix<binary_hv_buffer>::cpyFrom(matrix<float_2d_buffer> &input)
    {   
        size_t vec_size = input.hv_buffer.get_range()[1];

        if (vec_size / 8 != this->hv_buffer.get_range()[1])
            throw std::invalid_argument("Float matrix not compatible (input vector size / 8 != this vector size)");

        this->q.submit([&](cl::sycl::handler &h){
            auto float_acc = input.hv_buffer.get_access(h,cl::sycl::read_only);
            auto bin_acc = this->hv_buffer.get_access(h,cl::sycl::write_only, cl::sycl::no_init);
            h.parallel_for(this->hv_buffer.get_range(),[=](cl::sycl::id<2> lr){
                size_t i = lr[0];
                size_t j = lr[1];
                char v = 0;
                for (int N = 0; N<8; N++){
                    size_t j_ = j*8+N;
                    v |= (float_acc[i][j_] > 0) << N;
                }
                bin_acc[i][j].ch = v;
            });
        });

    }

}