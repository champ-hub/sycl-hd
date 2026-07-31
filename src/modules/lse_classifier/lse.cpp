/**
 * @file lse.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-10-15
 * 
 * @copyright GPL3
 * 
 */

#include "lse.hpp"
#include "lapack.h"
//#include "cblas.h"

// Make use of symmetric matrices (unstable)
// #define HD_LSE_SYM_METHOD

namespace hd::lse {

    classifier::classifier(cl::sycl::range<2> r, cl::sycl::queue &q_i) :
        q(q_i),
        weights(r),
        dim(r[0]),
        n_classes(r[1])
    {}

    classifier::classifier(const size_t nd, const size_t nc, cl::sycl::queue &q_i) :
        classifier(cl::sycl::range<2>(nd,nc), q_i) 
    {}

    void classifier::train_weights(hd::float_2d_buffer& inputs, hd::index_1d_buffer& labels) {
        auto targets = labelidxs_to_targets(labels, n_classes, q);
        this->train_weights(inputs, targets);
    }

    void classifier::train_weights(hd::float_2d_buffer& inputs, hd::float_2d_buffer& targets) {
        weights = lse(inputs, targets, q, this->g_i);
    }

    hd::float_2d_buffer classifier::sim_mat(hd::float_2d_buffer& inputs) {
        return mult(inputs, weights, q);
    }

    hd::index_1d_buffer classifier::classify(hd::float_2d_buffer& inputs) {
        auto sm = this->sim_mat(inputs);
        return argmax(sm, q);
    }

    double classifier::test(hd::float_2d_buffer& inputs, hd::index_1d_buffer& labels) {
        const size_t n_entries = inputs.get_range()[0];

        auto idxs = this->classify(inputs);
        auto o = hd::int_1d_buffer(n_entries);

        q.submit([&](cl::sycl::handler &h) {
            auto labels_acc = labels.get_access(h,cl::sycl::read_only);
            auto idxs_acc = idxs.get_access(h,cl::sycl::read_only);
            auto o_acc = o.get_access(h,cl::sycl::write_only, cl::sycl::no_init);
            const auto nc = n_classes;
            h.parallel_for(o.get_range()[0], [=](cl::sycl::id<1> lr) {
                o_acc[lr] = (labels_acc[lr] == idxs_acc[lr]) ? 1 : 0;
            });
        });

        int sum = 0;
        auto o_acc = o.get_host_access(cl::sycl::read_only);
        for(auto &v: o_acc)
            sum += v;

        return sum / (double) n_entries;

    }

}

namespace hd::lse {

    hd::float_2d_buffer labelidxs_to_targets(hd::index_1d_buffer& labels, const size_t n_classes, cl::sycl::queue &q) {

        const size_t n_entries = labels.get_range()[0];
        const cl::sycl::range r(n_entries, n_classes);

        hd::float_2d_buffer outputs(r);
        outputs.set_write_back(false);

        // Init output to 0s
        q.submit([&](cl::sycl::handler &h) {
            auto outputs_acc = outputs.get_access(h,cl::sycl::write_only, cl::sycl::no_init);
            h.parallel_for(r, [=](cl::sycl::id<2> lr) {
                outputs_acc[lr] = 0.0f;
            });
        });

        // Set correct outputs to 1
        q.submit([&](cl::sycl::handler &h) {
            auto targets_acc = labels.get_access(h,cl::sycl::read_only);
            auto outputs_acc = outputs.get_access(h,cl::sycl::write_only);
            const auto nc = n_classes;
            h.parallel_for(n_entries, [=](cl::sycl::id<1> lr) {
                outputs_acc[lr][targets_acc[lr]] = 1.0f;
            });
        });

        return outputs;
    }


    hd::float_2d_buffer projection(hd::float_2d_buffer& H, hd::float_2d_buffer& Y, cl::sycl::queue &q){
        cl::sycl::range<2> out_r(H.get_range()[1], Y.get_range()[1]);
        hd::float_2d_buffer C(out_r);

        q.submit([&](cl::sycl::handler &h) {
            auto H_acc = H.get_access(h,cl::sycl::read_only);
            auto Y_acc = Y.get_access(h,cl::sycl::read_only);
            auto C_acc = C.get_access(h,cl::sycl::write_only, cl::sycl::no_init);
            h.parallel_for(out_r, [=](cl::sycl::id<2> lr) {
                float sum = 0.0f;
                const size_t i = lr[0];
                const size_t j = lr[1];
                for (size_t k = 0; k < H.get_range()[0]; k++) {
                    sum += H_acc[k][i] * Y_acc[k][j];
                }
                C_acc[lr] = sum;
            });
        });
        return C;
    }

    void plus_i(hd::float_2d_buffer& C, cl::sycl::queue &q, const float i){
        q.submit([&](cl::sycl::handler &h) {
            auto C_acc = C.get_access(h,cl::sycl::read_write);
            h.parallel_for(C.get_range()[0], [=](cl::sycl::id<1> j) {
                C_acc[j][j] += i;
            });
        });
    }

    hd::float_2d_buffer corr_matrix(hd::float_2d_buffer& H, cl::sycl::queue &q){
        return projection(H, H, q);
    }

    hd::float_2d_buffer mult(hd::float_2d_buffer &A, hd::float_2d_buffer &B, cl::sycl::queue &q){
        cl::sycl::range<2> out_r(A.get_range()[0], B.get_range()[1]);
        hd::float_2d_buffer C(out_r);

        q.submit([&](cl::sycl::handler &h) {
            auto A_acc = A.get_access(h,cl::sycl::read_only);
            auto B_acc = B.get_access(h,cl::sycl::read_only);
            auto C_acc = C.get_access(h,cl::sycl::write_only, cl::sycl::no_init);
            h.parallel_for(out_r, [=](cl::sycl::id<2> lr) {
                float sum = 0.0f;
                const size_t i = lr[0];
                const size_t j = lr[1];
                for (size_t k = 0; k < A.get_range()[1]; k++) {
                    sum += A_acc[i][k] * B_acc[k][j];
                }
                C_acc[lr] = sum;
            });
        });
        return C;
    }

    hd::float_2d_buffer weight_mat(hd::float_2d_buffer& sym_inv_cor_mat, hd::float_2d_buffer proj_mat, cl::sycl::queue &q){
        const size_t dim = sym_inv_cor_mat.get_range()[0];
        const size_t n_classes = proj_mat.get_range()[1];

        cl::sycl::range<2> out_r(dim, n_classes);
        hd::float_2d_buffer C(out_r);

        q.submit([&](cl::sycl::handler &h) {
            auto A_acc = sym_inv_cor_mat.get_access(h,cl::sycl::read_only);
            auto B_acc = proj_mat.get_access(h,cl::sycl::read_only);
            auto C_acc = C.get_access(h,cl::sycl::write_only, cl::sycl::no_init);
            h.parallel_for(out_r, [=](cl::sycl::id<2> lr) {
                size_t i = lr[0];
                size_t j = lr[1];

                float sum = 0.0f;
                for (size_t k = 0; k < i; k++) {
                    sum += A_acc[k][i] * B_acc[k][j];
                }
                for (size_t k = i; k < dim; k++) {
                    sum += A_acc[i][k] * B_acc[k][j];
                }
                C_acc[lr] = sum;
            });
        });
        return C;
    }

    /**
     * @brief General matrix inversion
     * 
     * @param A matrix to be inverted (in/out)
     */
    void inv(hd::float_2d_buffer &A){
        const auto range = A.get_range();
        const int N = range[0];

        auto A_acc = A.get_host_access(cl::sycl::read_write);
        float* A_P = A_acc.get_pointer();

        // Could use svd since symetric matrix
        int *IPIV = new int[N];
        int LWORK = N*N;
        float *WORK = new float[LWORK];
        int INFO;

        sgetrf_(&N,&N,A_P,&N,IPIV,&INFO);
        if (INFO) throw std::runtime_error("Correlation Matrix is not invertible!");
        sgetri_(&N,A_P,&N,IPIV,WORK,&LWORK,&INFO);

        delete[] IPIV;
        delete[] WORK;
    }

    /**
     * @brief Symmetric matrix inversion with invertible check
     *  RETURNS A LOWER TRIANGULAR MATRIX!
     * @param A matrix to be inverted (in/out)
     */
    void inv_sym(hd::float_2d_buffer &A){
        const auto range = A.get_range();
        const int N = range[0];

        auto A_acc = A.get_host_access(cl::sycl::read_write);
        float* A_P = A_acc.get_pointer();

        int INFO;
        const char uplo = 'L';
        
        // Cholesky Factorization
        LAPACK_spotrf(&uplo, &N, A_P, &N, &INFO);
        if (INFO) throw std::runtime_error("Correlation Matrix is not invertible!");
        LAPACK_spotri(&uplo, &N, A_P, &N, &INFO);

    }

    hd::float_2d_buffer lse(hd::float_2d_buffer &H, hd::float_2d_buffer &Y, cl::sycl::queue &q, const bool sum_i){        // Correlation Matrix
        auto corr_mat = corr_matrix(H, q);

        if (sum_i) plus_i(corr_mat,q);

        // Projection Matrix
        auto proj_mat = projection(H, Y, q);

#ifndef HD_LSE_SYM_METHOD
        // Method 1
        inv(corr_mat);
        return mult(corr_mat, proj_mat, q);
#else
        //Method 2
        inv_sym(corr_mat); // LOWER TRIANGULAR MATRIX!
        return weight_mat(corr_mat, proj_mat, q); //Lower TRI * General MAT
#endif

    }

    template<typename acc>
    constexpr int argmax(acc &sims, const int entry, const int n_classes) {
        float max = sims[entry][0];
        int max_idx = 0;
        for (int i = 1; i < n_classes; ++i) {
            if (sims[entry][i] > max) {
                max = sims[entry][i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    hd::index_1d_buffer argmax(hd::float_2d_buffer &A, cl::sycl::queue &q){
        const size_t n_entries = A.get_range()[0];
        const size_t n_classes = A.get_range()[1];

        hd::index_1d_buffer res(n_entries);
        q.submit([&](cl::sycl::handler &h) {
            auto A_acc = A.get_access(h,cl::sycl::read_only);
            auto res_acc = res.get_access(h,cl::sycl::write_only, cl::sycl::no_init);
            h.parallel_for(n_entries, [=](cl::sycl::id<1> lr) {
                res_acc[lr] = argmax(A_acc, lr, n_classes);
            });
        });
        return res;
    }

}