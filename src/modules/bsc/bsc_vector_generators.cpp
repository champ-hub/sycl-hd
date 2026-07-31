/**
 * @file vector_generators.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-06
 * 
 * @copyright GPL3
 * 
 */

#include "bsc.hpp"
#include "../../core/macros.hpp"

static const hd::b8_char b8_all_true(1);
static const hd::b8_char b8_all_false(0);

namespace hd {// Helper functions CONSTANT VEC GENERATION

    void constantVectorGenerator(binary_hv_buffer &buffer, const b8_char value, cl::sycl::queue &q) {
        q.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc(buffer, h, cl::sycl::write_only, cl::sycl::no_init);
            h.parallel_for(buffer.get_range(), [=](cl::sycl::id<2> i) {
                acc[i[0]][i[1]] = value;
            });
        });
    }
    
}

namespace hd {// Helper functions RANDOM DENSE BINARY GENERATION

    RANDOM_DENSE_BIN(binary_hv_buffer,{
        char v = 0;
        for (ushort b = 0; b < 8; b++)
            v = v | hd::random01() << b; // set bit b
        v_e.ch = v;
    })


}

namespace hd {// Helper functions LEVEL VECTORS GENERATOR

    template <typename buff_type = binary_hv_buffer>
    void _levelVectorGenerator(buff_type &buffer, const bool full_level, const char inv_char, cl::sycl::queue &q) {
        
        size_t vector_size = buffer.get_range()[1];
        size_t n_vecs2gen = buffer.get_range()[0];

        cl::sycl::range<2> t_buff_range(1,vector_size);
        buff_type t_buffer(t_buff_range);

        randomDenseBinVectorGenerator(t_buffer);


        q.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc(buffer, h, cl::sycl::write_only, cl::sycl::no_init);
            cl::sycl::accessor acc_t(t_buffer, h, cl::sycl::read_only);
            h.parallel_for(n_vecs2gen, [=](cl::sycl::id<1> i) {
                int full_level_factor = (!full_level) + 1;
                uint n_pos_to_toggle = (i*vector_size)/(n_vecs2gen*full_level_factor);
                for (uint j = 0; j < n_pos_to_toggle ;j++)
                    acc[i][j] = acc_t[0][j];
                for (uint j = n_pos_to_toggle; j < vector_size ;j++)
                    acc[i][j].ch = acc_t[0][j].ch ^ inv_char;
            });
        });
    }

/*     template <typename buff_type = int_2d_buffer>
    void levelVectorGenerator(buff_type &buffer, const bool full_level, cl::sycl::queue &q) {

        uint vector_size = buffer.get_range()[1];

        cl::sycl::range<2> t_buff_range(1,vector_size);
        buff_type t_buffer(t_buff_range);

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

    } */
    
    void levelVectorGenerator(binary_hv_buffer &buffer, const bool full_level, cl::sycl::queue &q) {
        _levelVectorGenerator(buffer, full_level, 0b11111111, q);
    }

}

namespace hd {// Helper functions CIRCULAR VECTORS GENERATOR

    template <typename buff_type>
    void _circularVectorGenerator(buff_type &buffer, const char inv_char, cl::sycl::queue &q) {
        size_t n_vecs2gen = buffer.get_range()[0];
        size_t vector_size = buffer.get_range()[1];

        cl::sycl::range<2> t_buff_range(1,vector_size);
        buff_type t_buffer(t_buff_range);
        t_buffer.set_write_back(false);

        randomDenseBinVectorGenerator(t_buffer);

        q.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc(buffer, h, cl::sycl::write_only, cl::sycl::no_init);
            cl::sycl::accessor acc_t(t_buffer, h, cl::sycl::read_only);
            h.parallel_for(n_vecs2gen/2, [=](cl::sycl::id<1> i) {
                uint n_pos_to_toggle = (i*vector_size*2)/n_vecs2gen;
                for (uint j = 0; j < n_pos_to_toggle ;j++)
                    acc[i][j].ch = acc_t[0][j].ch ^ inv_char;
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
                    acc[n_vecs2gen/2+i][j_].ch = acc_t[0][j_].ch ^ inv_char;
                }
            });
        });
    }

    void circularVectorGenerator(binary_hv_buffer &buffer, cl::sycl::queue &q) {
        _circularVectorGenerator(buffer,0b11111111,q);

    }

}

namespace hd { // matrix

    template<>
    void matrix<binary_hv_buffer>::generateVectors(vectors_generator gen){
        switch (gen) {
            case all_zero:
                throw std::invalid_argument("BSC model does not have an 'all_zero' generation, did you mean 'all_false'?");
            case all_false:
                return constantVectorGenerator(this->hv_buffer,b8_all_false,this->q);
            case all_true:
                return constantVectorGenerator(this->hv_buffer,b8_all_true,this->q);
            case random:
                return randomDenseBinVectorGenerator(this->hv_buffer);
            case half_level:
                return levelVectorGenerator(this->hv_buffer,false,this->q);
            case full_level:
                return levelVectorGenerator(this->hv_buffer,true,this->q);
            case circular:
                return circularVectorGenerator(this->hv_buffer, this->q);
            case none:
                break;  
        }
    }

}

namespace hd { // INSTANTIATION
    template class matrix<binary_hv_buffer>;
}