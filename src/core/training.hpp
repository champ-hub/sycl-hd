/**
 * @file training.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief Training and retraining helper functions
 * @version 0.1
 * @date 2024-02-09
 *
 *
 * @copyright GPL3
 * 
 */

#ifndef SYCLHD_TRAINING_HPP
#define SYCLHD_TRAINING_HPP

#include "utilities.hpp"

namespace hd { //Process Reduce Labels Function

    template <typename label_type>
    struct p_labels {
        std::vector<label_type> unique_labels;
        index_1d_buffer label_nidx_corr;
        op_indexes op_indexes;
    };

    template <typename label_type>
    p_labels<label_type> _processReduceLabels(std::vector<label_type> const& provided_labels){
        SYCL_HD_PRINT("processReduceLabels(): Processing " << (provided_labels.size()) << " labels");

        std::vector<label_type> unique_labels;
        index_1d_buffer label_nidx_corr(provided_labels.size());
        label_nidx_corr.set_write_back(false);

        std::vector<std::vector<uint>> occurences_v = {};

        {
            auto lnic = label_nidx_corr.get_host_access(cl::sycl::write_only, cl::sycl::no_init);
            for (unsigned int i = 0; i < provided_labels.size(); i++) {
                auto iterator = std::find(unique_labels.begin(), unique_labels.end(), provided_labels[i]);
                if (iterator == unique_labels.end()) {
                    unique_labels.push_back(provided_labels[i]);
                    occurences_v.push_back({});
                    iterator = unique_labels.end()-1;
                }
                occurences_v[iterator - unique_labels.begin()].push_back(i);
                SYCL_HD_PRINT("processReduceLabels(): Correspondence from " << (i) << " to " << (iterator - unique_labels.begin()));
                lnic[i] = iterator - unique_labels.begin();
            }
        }

        auto op = imat2op(occurences_v,provided_labels.size());

        return {unique_labels, label_nidx_corr, op};
    }

    template <typename label_type>    
    index_1d_buffer labels2indexes(const std::vector<label_type>& labels, const std::vector<label_type>& unique_labels){

            const size_t n_entries = labels.size();

            index_1d_buffer ret(n_entries);
            ret.set_write_back(false);
            auto ret_acc = ret.get_host_access(cl::sycl::write_only,cl::sycl::no_init);

            for (unsigned int i = 0; i < n_entries; i++) {
                auto iterator = std::find(unique_labels.begin(), unique_labels.end(), labels[i]);
                
                if (iterator == unique_labels.end()) 
                    throw std::runtime_error(
                        "labels2indexes(): provided label at position " + 
                        std::to_string(i) +
                        " not found in this->unique_labels");
                
                 ret_acc[i] = iterator - unique_labels.begin();
            }

            return ret;
        }

}


namespace hd { // Retraining Iteration functions
    
    int retrainIterationVoiceHD(
        float_2d_buffer &accumulator,
        float_2d_buffer &test_entries,
        float_2d_buffer &distance_matrix, 
        index_1d_buffer &idx_ans,
        cl::sycl::queue &q);

    int retrainIterationAdaptHD(
        float_2d_buffer &accumulator,
        float_2d_buffer &test_entries,
        float_2d_buffer &distance_matrix, 
        index_1d_buffer &idx_ans,
        cl::sycl::queue &q,
        const float alpha = 2.5
    );
    
    int retrainIterationOnlineHD(
        float_2d_buffer &accumulator,
        float_2d_buffer &test_entries, 
        float_2d_buffer &distance_matrix, 
        index_1d_buffer &idx_ans,
        cl::sycl::queue &q,
        const float alpha = 2.5
    );
}

namespace hd { // Regeneration helpers
    index_1d_buffer selectIndexes(float_1d_buffer &var_vec, const float ratio = 0.1);
}



#endif //SYCLHD_TRAINING_HPP