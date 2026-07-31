/**
 * @file tmap_training.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-11
 * 
 * @copyright GPL3
 * 
 */

#include "tmap.hpp"

namespace hd { // Convert to and from float matrix

    template<>
    matrix<float_2d_buffer> matrix<ternary_hv_buffer>::toFloatMatrix()
    {
        cl::sycl::range r(this->hv_buffer.get_range()[0],this->hv_buffer.get_range()[1]*4);
        matrix<float_2d_buffer> to_ret(r,this->q);

        q.submit([&](cl::sycl::handler &h){
            auto ternary_acc = this->hv_buffer.get_access(h,cl::sycl::read_only);
            auto float_acc = to_ret.hv_buffer.get_access(h,cl::sycl::write_only, cl::sycl::no_init);
            h.parallel_for(this->hv_buffer.get_range(),[=](cl::sycl::id<2> lr){
                size_t i = lr[0];
                size_t j = lr[1];

                float_acc[i][j*4] = ternary_acc[i][j].t0;
                float_acc[i][j*4+1] = ternary_acc[i][j].t1;
                float_acc[i][j*4+2] = ternary_acc[i][j].t2;
                float_acc[i][j*4+3] = ternary_acc[i][j].t3;
            });
        });

        return to_ret;
    }

    template<>
    void matrix<ternary_hv_buffer>::cpyFrom(matrix<float_2d_buffer> &input)
    {   
        size_t vec_size = input.hv_buffer.get_range()[1];

        if (vec_size / 4 != this->hv_buffer.get_range()[1])
            throw std::invalid_argument("Float matrix not compatible (input vector size / 4 != this vector size)");

        this->q.submit([&](cl::sycl::handler &h){
            auto float_acc = input.hv_buffer.get_access(h,cl::sycl::read_only);
            auto ternary_acc = this->hv_buffer.get_access(h,cl::sycl::write_only, cl::sycl::no_init);
            h.parallel_for(this->hv_buffer.get_range(),[=](cl::sycl::id<2> lr){
                size_t i = lr[0];
                size_t j = lr[1];

                t4_ints v = 0;

                v.t0 = float_acc[i][j*4];
                v.t1 = float_acc[i][j*4+1];
                v.t2 = float_acc[i][j*4+2];
                v.t3 = float_acc[i][j*4+3];

                ternary_acc[i][j] = v.normalize();
            });
        });

    }
}