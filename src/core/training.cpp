/**
 * @file training.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief Training methods
 * @version 0.1
 * @date 2024-02-09
 * 
 * TODO: Retraining Iteration: Start with VoiceHD
 *
 * @copyright GPL3
 * 
 */

#include "training.hpp"
#include "query.hpp"

namespace hd {

    /**
     * @brief Retrain Step for VoiceHD Retraining Style
     *
     * Correct Class += Missclassified HV
     * Wrong Class -= Missclassified HV
     * 
     * @param accumulator Accumulator input and result
     * @param test_entries HV entries
     * @param distance_matrix Distance matrix of each entry to the classes
     * @param idx_ans Correct Indexes that entries should obtain
     * @param q SYCL q
     *
     * @return Missclassifications number
     */
    int retrainIterationVoiceHD(
        float_2d_buffer &accumulator,
        float_2d_buffer &test_entries,
        float_2d_buffer &distance_matrix,
        index_1d_buffer  &idx_ans,
        cl::sycl::queue &q
    ) {

        int missclassifications = 0;

        index_1d_buffer idxs =
            idxOfHighestValueInMatRow(distance_matrix, q);
        idxs.set_write_back(false);

        auto ans = idx_ans.get_host_access(cl::sycl::read_only);
        auto ret = idxs.get_host_access(cl::sycl::read_only);
        for (int i = 0; i < idx_ans.get_range()[0]; i++){
            if (ans[i] != ret[i]){
                missclassifications++;
                size_t correct_i = ans[i];
                size_t wrong_i = ret[i];
                q.submit([&](cl::sycl::handler &h){
                    auto accum_acc = accumulator.get_access(h,cl::sycl::read_write);
                    auto entry_acc = test_entries.get_access(h,cl::sycl::read_only);
                    h.parallel_for(accumulator.get_range()[1],[=](cl::sycl::id<1> dim){
                        accum_acc[correct_i][dim] += entry_acc[i][dim]; //Add influence to correct class
                        accum_acc[wrong_i][dim] -= entry_acc[i][dim];   //Remove influence from wrong class
                    });
                });
            }
        }
        return missclassifications;
    }

    /**
     * @brief Retrain Step for AdaptHD Retraining Style
     *
     * Correct Class += Missclassified HV * alpha
     * Wrong Class -= Missclassified HV * alpha
     * 
     * @param accumulator Accumulator input and result
     * @param test_entries HV entries
     * @param distance_matrix Distance matrix of each entry to the classes
     * @param idx_ans Correct Indexes that entries should obtain
     * @param q SYCL q
     * @param alpha factor
     *
     * @return Missclassifications number
     */
    int retrainIterationAdaptHD(
        float_2d_buffer &accumulator,
        float_2d_buffer &test_entries, 
        float_2d_buffer &distance_matrix, 
        index_1d_buffer &idx_ans,
        cl::sycl::queue &q,
        const float alpha
    ) {
        int missclassifications = 0;

        index_1d_buffer idxs =
            idxOfHighestValueInMatRow(distance_matrix, q);
        idxs.set_write_back(false);

        auto ans = idx_ans.get_host_access(cl::sycl::read_only);
        auto ret = idxs.get_host_access(cl::sycl::read_only);
        for (int i = 0; i < idx_ans.get_range()[0]; i++){
            if (ans[i] != ret[i]){
                missclassifications++;
                size_t correct_i = ans[i];
                size_t wrong_i = ret[i];
                q.submit([&](cl::sycl::handler &h){
                    auto accum_acc = accumulator.get_access(h,cl::sycl::read_write);
                    auto entry_acc = test_entries.get_access(h,cl::sycl::read_only);
                    h.parallel_for(accumulator.get_range()[1],[=](cl::sycl::id<1> dim){
                        accum_acc[correct_i][dim] += alpha*entry_acc[i][dim]; //Add influence to correct class
                        accum_acc[wrong_i][dim] -= alpha*entry_acc[i][dim];   //Remove influence from wrong class
                    });
                });
            }
        }
        return missclassifications;
    }

    /**
     * @brief Retrain Step for OnlineHD Retraining Style
     *
     * Correct class += Missclassified HV*alpha*(1-sim(Correct Class, Miss. HV))
     * Wrong class -= Missclassified HV*alpha*(1-sim(Wrong Class, Miss. HV))
     * 
     * @param accumulator Accumulator input and result
     * @param test_entries HV entries
     * @param distance_matrix Distance matrix of each entry to the classes
     * @param idx_ans Correct Indexes that entries should obtain
     * @param q SYCL q
     * @param alpha factor
     *
     * @return Missclassifications number
     */
    int retrainIterationOnlineHD(
        float_2d_buffer &accumulator,
        float_2d_buffer &test_entries, 
        float_2d_buffer &distance_matrix, 
        index_1d_buffer &idx_ans,
        cl::sycl::queue &q,
        const float alpha
    ) {

        int missclassifications = 0;

        auto idxs =
            idxOfHighestValueInMatRow(distance_matrix, q);
        
        auto ans = idx_ans.get_host_access(cl::sycl::read_only);
        auto ret = idxs.get_host_access(cl::sycl::read_only);
        auto distm_acc = distance_matrix.get_host_access(cl::sycl::read_only);
        for (int i = 0; i < ans.get_range()[0]; i++)
            if (ans[i] != ret[i]){
                missclassifications++;

                size_t correct_i = ans[i];
                size_t wrong_i = ret[i];

                float c_corr_factor = (1-distm_acc[i][correct_i])*alpha; // Correct class correction factor
                float w_corr_factor = (1-distm_acc[i][wrong_i]  )*alpha; // Wrong class correction factor

                q.submit([&](cl::sycl::handler &h){
                    auto accum_acc = accumulator.get_access(h,cl::sycl::read_write);
                    auto entry_acc = test_entries.get_access(h,cl::sycl::read_only);
                    h.parallel_for(accumulator.get_range()[1],[=](cl::sycl::id<1> dim){
                        accum_acc[correct_i][dim] += entry_acc[i][dim]*c_corr_factor; //Add influence to correct class
                        accum_acc[wrong_i][dim] -= entry_acc[i][dim]*w_corr_factor;   //Remove influence from wrong class
                    });
                });
            }
        return missclassifications;
    }
}

namespace hd {

    /**
     * @brief A function to select indexes based on a given ratio from a 1D buffer of floats.
     *  index of the (ratio*100)% lowest valued members are selected
     *
     * TODO: Study the possibility of parallelizing this
     *
     * @param var_vec the 1D buffer of floats
     * @param ratio the ratio used to select indexes
     *
     * @return a 1D buffer of indexes
     */
    index_1d_buffer selectIndexes(float_1d_buffer &var_vec, const float ratio) {

        auto acc = var_vec.get_host_access(cl::sycl::read_only);
        size_t sz = var_vec.get_range()[0];

        std::vector<uint> idxs(sz);
        std::iota(idxs.begin(),idxs.end(),0);

        auto compare = [&](size_t i1, size_t i2){
            return acc[i1] < acc[i2];   
        };

        size_t cutoff_index = static_cast<size_t>(sz * ratio);

        std::partial_sort(idxs.begin(),idxs.begin()+cutoff_index,idxs.end(),compare);

        return index_1d_buffer{idxs.begin(),idxs.begin()+cutoff_index};

    }

}