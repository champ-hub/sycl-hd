/**
 * @file lse.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief Least square error classification HDC
 * @version 0.1
 * @date 2024-10-15
 * 
 * @copyright GPL3
 * 
 */
#include "../../core/definitions.hpp"
#include "../../core/training.hpp"
#include "../../core/matrix.hpp"

#ifndef HDC_LSE_HPP
#define HDC_LSE_HPP

namespace hd::lse {
    /**
     * @brief VSLSC - LSE Classifier object
     * 
     */
    class classifier {
      public:
        hd::float_2d_buffer weights; ///< Weights computed by LSE
        cl::sycl::queue &q;
        const size_t n_classes;
        const size_t dim; ///< Hypervector Dimensionality
        bool g_i = false; ///< Guarantee inverse, adds an identity term to the correlation matrix

        classifier(const size_t nd, const size_t nc, cl::sycl::queue &q_i);
        classifier(cl::sycl::range<2> r, cl::sycl::queue &q_i);

        void train_weights(hd::float_2d_buffer& inputs, hd::index_1d_buffer& labels);
        void train_weights(hd::float_2d_buffer& inputs, hd::float_2d_buffer& targets);

        hd::float_2d_buffer sim_mat(hd::float_2d_buffer& inputs);
        hd::index_1d_buffer classify(hd::float_2d_buffer& inputs);
        double test(hd::float_2d_buffer& inputs, hd::index_1d_buffer& labels);

        template<typename buffer_type>
        void train_weights(hd::matrix<buffer_type> &inputs, hd::index_1d_buffer& labels){
            auto in = inputs.toFloatMatrix();
            return this->train_weights(in.hv_buffer, labels);
        }

        template<typename buffer_type>
        double test(hd::matrix<buffer_type> &inputs, hd::index_1d_buffer& labels){
            auto in = inputs.toFloatMatrix();
            return this->test(in.hv_buffer, labels);
        }

        void train_weights(hd::matrix<hd::float_2d_buffer> &inputs, hd::index_1d_buffer& labels){
            return this->train_weights(inputs.hv_buffer, labels);
        }

        double test(hd::matrix<hd::float_2d_buffer> &inputs, hd::index_1d_buffer& labels){
            return this->test(inputs.hv_buffer, labels);
        }

    };

    template<typename Tp>
    class l_classifier : public classifier {      
      public:
        const std::vector<Tp> unique_labels;

        l_classifier(const std::vector<Tp> &unique_labels_i, const size_t nd, cl::sycl::queue &q_i):
          classifier(nd, unique_labels_i.size(), q_i),
          unique_labels(unique_labels_i) {}
        
        using classifier::train_weights;
        using classifier::test;

        void train_weights(hd::float_2d_buffer& inputs, const std::vector<Tp>& labels){
            auto labels_i = hd::labels2indexes(labels, this->unique_labels);
            return this->train_weights(inputs, labels_i);
        }
        
        double test(hd::float_2d_buffer& inputs, const std::vector<Tp>& labels){
            auto labels_i = hd::labels2indexes(labels,this->unique_labels);
            return this->test(inputs, labels_i);
        }

        template<typename buffer_type>
        void train_weights(hd::matrix<buffer_type> &inputs, const std::vector<Tp>& labels){
            auto labels_i = hd::labels2indexes(labels,this->unique_labels);
            return this->train_weights(inputs, labels_i);
        }

        template<typename buffer_type>
        double test(hd::matrix<buffer_type> &inputs, const std::vector<Tp>& labels){
            auto labels_i = hd::labels2indexes(labels,this->unique_labels);
            return this->test(inputs, labels_i);
        }

    };

}

namespace hd::lse { // UTILITY
    hd::float_2d_buffer labelidxs_to_targets(hd::index_1d_buffer& labels, const size_t n_classes, cl::sycl::queue &q);
}


namespace hd::lse { // MATH
    hd::float_2d_buffer projection(hd::float_2d_buffer& H, hd::float_2d_buffer& Y, cl::sycl::queue &q);
    hd::float_2d_buffer corr_matrix(hd::float_2d_buffer& H, cl::sycl::queue &q);
    
    hd::float_2d_buffer weight_mat(hd::float_2d_buffer& sym_inv_cor_mat, hd::float_2d_buffer proj_mat, cl::sycl::queue &q);
    hd::float_2d_buffer mult(hd::float_2d_buffer &A, hd::float_2d_buffer &B, cl::sycl::queue &q);
    
    void inv(hd::float_2d_buffer &A);
    void inv_sym(hd::float_2d_buffer &A);
    void plus_i(hd::float_2d_buffer& H, cl::sycl::queue &q, const float i = 1);

    hd::float_2d_buffer lse(hd::float_2d_buffer &H, hd::float_2d_buffer &Y, cl::sycl::queue &q, const bool sum_i = false);
    hd::index_1d_buffer argmax(hd::float_2d_buffer &A, cl::sycl::queue &q);
}




#endif //HDC_LSE_HPP