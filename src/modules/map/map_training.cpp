/**
 * @file map_training.cpp
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

    template<>
    matrix<float_2d_buffer> matrix<float_2d_buffer>::toFloatMatrix(){
        std::cout << "WARNING: Convert float matrix to float matrix, refactor the method that caused this" << std::endl;
        return *this;
    }

    template<>
    void matrix<float_2d_buffer>::cpyFrom(matrix<float_2d_buffer> &input){
        this->hv_buffer = input.hv_buffer;
    }

    void _normalize(float_2d_buffer &buff, cl::sycl::queue &q){

        auto r = buff.get_range();
        float_1d_buffer l(r[0]);

        q.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc(buff, h, cl::sycl::read_only);
            cl::sycl::accessor acc_l(l, h, cl::sycl::write_only, cl::sycl::no_init);
            h.parallel_for(r[0], [=](cl::sycl::id<1> i) {
                float l = 0;
                for (int j = 0; j < r[1]; j++)
                    l += acc[i][j]*acc[i][j];
                acc_l[i] = cl::sycl::sqrt(l);
            });
        });

        q.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc(buff, h, cl::sycl::read_write);
            cl::sycl::accessor acc_l(l, h, cl::sycl::read_only);
            h.parallel_for(r, [=](cl::sycl::id<2> i) {
                auto old = acc[i[0]][i[1]];
                acc[i[0]][i[1]] = old / acc_l[i[0]];
            });
        });
    }

    template<>
    void matrix<float_2d_buffer>::normalize(){
        _normalize(this->hv_buffer, this->q);
    }

    template<>
    matrix<float_2d_buffer> matrix<float_2d_buffer>::_train(
        op_indexes &occurences, 
        index_1d_buffer &class_idx, 
        const int rt_steps, 
        const float alpha,
        const retrain method
    ) {
        matrix<float_2d_buffer> accumulator = this->bundle(occurences);
        cl::sycl::range<2> r(accumulator.hv_buffer.get_range()[0], this->hv_buffer.get_range()[1]);

        int l_misses = 0; // Misses in last iteration
        int stagnation = 0; // Stagnation counter

        SYCL_HD_PRINT("_train(): ")

        for (int rt_step = 1; rt_step <= rt_steps; rt_step++){
            auto distance_matrix = accumulator.distanceMatrix(*this);
            int misses = 0;
            switch (method){
                case retrain::voicehd:
                    {misses = retrainIterationVoiceHD(accumulator.hv_buffer, this->hv_buffer, distance_matrix, class_idx, this->q);
                    break;}
                case retrain::onlinehd:
                    {misses = retrainIterationOnlineHD(accumulator.hv_buffer, this->hv_buffer, distance_matrix, class_idx, this->q,alpha);
                    break;}
                case retrain::adapthd:
                    {misses = retrainIterationAdaptHD(accumulator.hv_buffer, this->hv_buffer, distance_matrix, class_idx, this->q,alpha);
                    break;}
            }

            SYCL_HD_PRINT("     Iteration: " << rt_step)
            SYCL_HD_PRINT("     Misses: " << misses)
            
            if (misses == l_misses) stagnation++;
            else stagnation = 0;

            if (misses == 0 || stagnation > 3) break;

            l_misses = misses;
        }
        return accumulator;
    }
}