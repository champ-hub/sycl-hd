/**
 * @file utilities.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief Useful generic functions implementation 
 * @version 0.1
 * @date 2024-05-19
 * 
 * @copyright GPL3
 * 
 */

#include "utilities.hpp"
//#include <random>

namespace hd { // RANDOM

    /* [DISABLED]
    int random01() {
        static std::random_device dev;
        static std::mt19937 rng(dev());
        static std::bernoulli_distribution gen;
        return gen(rng);
    } */

    /**
     * @brief Fast Random Bernoulli Distribution
     * 
     * @return int 
     */
    int random01(){
        return rand() % 2; 
    }

    int random1sign(){
        return random01()*2 - 1; 
    }

}

namespace hd {
    op_indexes imat2op(std::vector<std::vector<uint>> &idxs, uint t_size)
    {
        index_1d_buffer occurences(t_size);
        index_1d_buffer new_idx(idxs.size()+1);

        {
            auto occ_a = occurences.get_host_access(cl::sycl::write_only, cl::sycl::no_init);
            auto nix_a = new_idx.get_host_access(cl::sycl::write_only, cl::sycl::no_init);
            nix_a[0] = 0;

            auto _i = 0;
            for (unsigned int i = 0; i < idxs.size(); i++) {
                for (unsigned int j = 0; j < idxs[i].size(); j++) {
                    occ_a[_i] = idxs[i][j];
                    _i++;
                }
                nix_a[i+1] = idxs[i].size() + nix_a[i];
            }
        }

        occurences.set_write_back(false);
        new_idx.set_write_back(false);

        return {occurences, new_idx};
    }

    op_indexes imat2op(std::vector<std::vector<uint>> &idxs){
        uint s = 0;
        for (std::vector<uint> v : idxs)
            s += v.size();
        return imat2op(idxs, s);
    }

    op_indexes bf2d2op(hd::index_2d_buffer &&buffer){

        auto r = buffer.get_range();
        size_t t_size = r[0]*r[1];

        index_1d_buffer occurences(t_size);
        index_1d_buffer new_idx(r[0]+1);

        {
            auto idxs = buffer.get_host_access(cl::sycl::read_only);
            auto occ_a = occurences.get_host_access(cl::sycl::write_only, cl::sycl::no_init);
            auto nix_a = new_idx.get_host_access(cl::sycl::write_only, cl::sycl::no_init);
            nix_a[0] = 0;

            auto _i = 0;
            for (unsigned int i = 0; i < r[0]; i++) {
                for (unsigned int j = 0; j < r[1]; j++) {
                    occ_a[_i] = idxs[i][j];
                    _i++;
                }
                nix_a[i+1] = r[1] + nix_a[i];
            }
        }

        occurences.set_write_back(false);
        new_idx.set_write_back(false);

        return {occurences, new_idx};
    }

}