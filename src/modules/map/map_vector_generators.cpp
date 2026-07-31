/**
 * @file map_vector_generators.cpp
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

namespace hd {// Helper functions CONSTANT VEC GENERATION

    void constantVectorGenerator(float_2d_buffer &buffer, float value, cl::sycl::queue &q) {
        q.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc(buffer, h, cl::sycl::write_only, cl::sycl::no_init);
            h.parallel_for(buffer.get_range(), [=](cl::sycl::id<2> i) {
                acc[i[0]][i[1]] = value;
            });
        });
    }
    
}

namespace hd {// Helper functions RANDOM DENSE BINARY GENERATION

    RANDOM_DENSE_BIN(float_2d_buffer,{
        v_e = hd::random1sign();
        //v_e = rand() % 100 > 50 ? 1 : -1; // HDCC Method
    })

}

namespace hd {// Helper functions LEVEL VECTORS GENERATOR

    void levelVectorGenerator(float_2d_buffer &buffer, const bool full_level, cl::sycl::queue &q) {

        uint vector_size = buffer.get_range()[1];

        cl::sycl::range<2> t_buff_range(1,vector_size);
        float_2d_buffer t_buffer(t_buff_range);

        randomDenseBinVectorGenerator(t_buffer);

        size_t n_vecs2gen = buffer.get_range()[0];

        q.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc(buffer, h, cl::sycl::write_only, cl::sycl::no_init);
            cl::sycl::accessor acc_t(t_buffer, h, cl::sycl::read_only);
            h.parallel_for(n_vecs2gen, [=](cl::sycl::id<1> i) {
                int full_level_factor = (!full_level) + 1;
                uint n_pos_to_toggle = (i*vector_size)/(n_vecs2gen*full_level_factor);
                for (uint j = 0; j < n_pos_to_toggle ;j++)
                    acc[i][j] = acc_t[0][j]*-1;
                for (uint j = n_pos_to_toggle; j < vector_size ;j++)
                    acc[i][j] = acc_t[0][j];
            });
        });

    }

}

namespace hd {// Helper functions CIRCULAR VECTORS GENERATOR

    void circularVectorGenerator(float_2d_buffer &buffer, cl::sycl::queue &q) {
        size_t n_vectors = buffer.get_range()[0];
        size_t vector_size = buffer.get_range()[1];

        cl::sycl::range<2> t_buff_range(1,vector_size);
        float_2d_buffer t_buffer(t_buff_range);
        t_buffer.set_write_back(false);

        randomDenseBinVectorGenerator(t_buffer);

        size_t n_vecs2gen = buffer.get_range()[0];

        q.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc(buffer, h, cl::sycl::write_only, cl::sycl::no_init);
            cl::sycl::accessor acc_t(t_buffer, h, cl::sycl::read_only);
            h.parallel_for(n_vecs2gen/2, [=](cl::sycl::id<1> i) {
                uint n_pos_to_toggle = (i*vector_size*2)/n_vecs2gen;
                for (uint j = 0; j < n_pos_to_toggle ;j++)
                    acc[i][j] = acc_t[0][j]*-1;
                for (uint j = n_pos_to_toggle; j < vector_size ;j++)
                    acc[i][j] = acc_t[0][j];
            });
        });

        q.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc(buffer, h, cl::sycl::write_only);
            cl::sycl::accessor acc_t(t_buffer, h, cl::sycl::read_only);
            h.parallel_for(n_vecs2gen/2, [=](cl::sycl::id<1> i) {
                uint n_pos_to_toggle = (i*vector_size*2)/n_vecs2gen;
                for (uint j = 0; j < n_pos_to_toggle ;j++){
                    uint j_ = vector_size-j-1;
                    acc[n_vecs2gen/2+i][j_] = acc_t[0][j_];
                }
                for (uint j = n_pos_to_toggle; j < vector_size ;j++){
                    uint j_ = vector_size-j-1;
                    acc[n_vecs2gen/2+i][j_] = acc_t[0][j_]*-1;
                }
            });
        });
    }
}

namespace hd {// generateVectors() MAP specialization

    template<>
    void matrix<float_2d_buffer>::generateVectors(vectors_generator gen){
        switch (gen) {
            case none:
                return;
            case all_zero:
                return constantVectorGenerator(this->hv_buffer,0,this->q);
            case all_false:
                return constantVectorGenerator(this->hv_buffer,false,this->q);
            case all_true:
                return constantVectorGenerator(this->hv_buffer,true,this->q);
            case random:
                return randomDenseBinVectorGenerator(this->hv_buffer);
            case half_level:
                return levelVectorGenerator(this->hv_buffer,false,this->q);
            case full_level:
                return levelVectorGenerator(this->hv_buffer,true,this->q);
            case circular:
                return circularVectorGenerator(this->hv_buffer, this->q);  
        }
    }
}