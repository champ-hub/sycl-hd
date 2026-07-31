/**
 * @file query.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief Methods for similarity computation and querying
 * @version 2.0
 * @date 2024-01-31
 * 
 * @copyright GPL3
 * 
 */
#include "query.hpp"
#include "macros.hpp"

namespace hd {

    COSINE_DISTANCE_MATRIX(float_2d_buffer,{
        float v1 = acc_encoded_test_entries[i][k];
        float v2 = acc_model_entries[j][k];
        num += v1*v2;
        den1 += v1*v1;
        den2 += v2*v2;
    })

    /**
     * @brief Get the index of highest value in matrix row and corresponding value
     * 
     * @param sim_mat Similarity Matrix
     * @param q SYCL queue
     * @return std::pair<index1d_buffer,float_1d_buffer> 
     */
    idx_sim_pair getIdxSimVec(float_2d_buffer &sim_mat, cl::sycl::queue &q){
        
        size_t ret_range = sim_mat.get_range()[0];
        size_t n_comp = sim_mat.get_range()[1];

        // INITIALIZE RETURN BUFFER
        index_1d_buffer to_return_idx(ret_range);
        float_1d_buffer to_return_sim(ret_range);

        to_return_idx.set_write_back(false);
        to_return_sim.set_write_back(false);

        q.submit([&](cl::sycl::handler &h) {
            
            cl::sycl::accessor acc_sim_mat(sim_mat, h, cl::sycl::read_only);
            cl::sycl::accessor acc_to_ret(to_return_idx, h, cl::sycl::write_only,cl::sycl::no_init);
            cl::sycl::accessor acc_to_ret_sim(to_return_sim, h, cl::sycl::write_only,cl::sycl::no_init);

            h.parallel_for(ret_range, [=](cl::sycl::id<1> i) {
                unsigned short int j_ret = 0;
                float max = acc_sim_mat[i][0];
                for (unsigned short int j = 1; j < n_comp; j++)
                    if (max < acc_sim_mat[i][j]){
                        max = acc_sim_mat[i][j];
                        j_ret = j;
                    }
                acc_to_ret[i] = j_ret;
                acc_to_ret_sim[i] = max;
            });
        });

        return {to_return_idx,to_return_sim};
    }

    /**
     * @brief Get the index of highest value in matrix row 
     * 
     * @param sim_mat Similarity Matrix 
     * @param q SYCL queue 
     * @return index1d_buffer 
     */
    index_1d_buffer idxOfHighestValueInMatRow(float_2d_buffer &sim_mat, cl::sycl::queue &q){
        size_t ret_range = sim_mat.get_range()[0];
        size_t n_comp = sim_mat.get_range()[1];

        // INITIALIZE RETURN BUFFER
        cl::sycl::buffer<unsigned short int, 1> to_return(ret_range);

        q.submit([&](cl::sycl::handler &h) {
            
            cl::sycl::accessor acc_sim_mat(sim_mat, h, cl::sycl::read_only);
            cl::sycl::accessor acc_to_ret(to_return, h, cl::sycl::write_only,cl::sycl::no_init);

            h.parallel_for(ret_range, [=](cl::sycl::id<1> i) {
                unsigned short int j_ret = 0;
                float max = acc_sim_mat[i][0];
                for (unsigned short int j = 1; j < n_comp; j++)
                    if (max < acc_sim_mat[i][j]){
                        max = acc_sim_mat[i][j];
                        j_ret = j;
                    }
                acc_to_ret[i] = j_ret;
            });
        });

        return to_return;
    }

    void varianceNormalize(float_1d_buffer &unnormVar, const int n_vecs, cl::sycl::queue &q){

        q.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc(unnormVar, h, cl::sycl::read_write);
            h.parallel_for(unnormVar.get_range(), [=](cl::sycl::id<1> i) {
                acc[i] /= n_vecs;
            });
        });
    }

}
//syclhd