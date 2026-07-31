/**
 * @file utilities.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-01
 * 
 * @copyright GPL3
 * 
 */

#ifndef SYCLHD_UTILITES_HPP
#define SYCLHD_UTILITES_HPP

#include "definitions.hpp"

// Include IOManip for debug functions
#include <iomanip>


namespace hd { // vector of vectors to index_2d_buffer

    /**
     * @brief Converts a 2D vector into a SYCL buffer.
     *
     * @param mat reference to the 2D vector
     *
     * @return a SYCL buffer containing the data from the input vector
     *
     * @throws None
     */
    template <typename Tp1, typename Tp2 = Tp1>
    inline cl::sycl::buffer<Tp2,2> mat2RectBuff(std::vector<std::vector<Tp1>> &mat){
        cl::sycl::buffer<Tp2,2> to_ret(cl::sycl::range(mat.size(),mat[0].size()));
        auto acc = to_ret.get_host_access(cl::sycl::write_only, cl::sycl::no_init);
        for(int i = 0; i < mat.size(); i++)
            for(int j = 0; j < mat[i].size(); j++)
                acc[i][j] = mat[i][j];

        return to_ret;
    }

    /**
     * @brief A function that converts a 2D matrix of indexes into a vector of 1D index buffers.
     *
     * @param indexes a vector of vectors representing the 2D matrix of indexes
     * @param max_index the maximum index value allowed, defaults to -1
     *
     * @return a vector of 1D index buffers
     */
    template <typename Tp = int>
    std::vector<index_1d_buffer> mat2VecIdxBuff(std::vector<std::vector<Tp>> const& indexes, const size_t max_index = -1){
        std::vector<index_1d_buffer> n_indexes(indexes.size(), index_1d_buffer(0));

        for (auto i = 0; i< indexes.size(); i++){
            index_1d_buffer a(indexes[i].begin(), indexes[i].end());
            a.set_write_back(false);
            n_indexes[i] = a;
        }
        return n_indexes;
    }



    /**
     * @brief A template function to convert a matrix of indexes to an index buffer.
     *
     * @param indexes a 2D vector of indexes
     * @param max_index the maximum index value allowed
     *
     * @return an index buffer representing the input indexes
     *
     * @throws std::invalid_argument if the data is not square or exceeds the maximum index
     */
    template <typename Tp = int>
    index_2d_buffer mat2RectIdxBuff(std::vector<std::vector<Tp>> const& indexes, const size_t max_index = -1){
        cl::sycl::range<2> r(indexes.size(), indexes[0].size());
        index_2d_buffer n_indexes(r);
        n_indexes.set_write_back(false);

        auto acc = n_indexes.get_host_access(cl::sycl::write_only, cl::sycl::no_init);

        for (auto i = 0; i< indexes.size(); i++){
            if (indexes[i].size() != r[1])
                throw std::invalid_argument("Data is not square");
            for (auto j = 0; j < r[1]; j++){
                if (((uint) indexes[i][j]) >= max_index)
                    throw std::invalid_argument("Data excedes index");
                acc[i][j] = indexes[i][j];
            }
        }
        return n_indexes;
    }
}
namespace hd {

    /**
     * @brief A function that performs barrelshift on the input data to generate an index buffer.
     * Called automatically by the ngram function when the barrelshift option is set
     *
     * @param data a reference to a vector of vectors of type Tp, representing the input data
     * @param ngram the size of the ngram for barrelshifting
     * @param break_size the size at which to break the ngram operation, usually the number of HVs of the caller matrix
     *
     * @return an index buffer generated from the barrelshift operation
     *
     */
    template <typename Tp>
    index_2d_buffer barrelshift(std::vector<std::vector<Tp>> const& data, const size_t ngram, const size_t break_size = 28){

        size_t max_range = std::max_element(
                data.begin(),
                data.end(),
                [](const std::vector<Tp> &lhs, const std::vector<Tp> &rhs) -> bool {
                return lhs.size() < rhs.size();
            })->size();
        
        cl::sycl::range<2> ret_range(data.size(),(max_range-ngram+1)*ngram);
        
        index_2d_buffer to_ret(ret_range);
        to_ret.set_write_back(false);

        cl::sycl::host_accessor acc_ret(to_ret, cl::sycl::write_only, cl::sycl::no_init);

        for (int i = 0; i < data.size(); i++){
            auto sz = data[i].size();
            for (int j = 0; j < sz-ngram+1; j++)
                for (int i_ = 0; i_ < ngram; i_++)
                    acc_ret[i][j*ngram+i_] = data[i][j+i_];
            acc_ret[i][(sz-ngram+1)*ngram] = break_size;
            // The rest of the buffer can be left unitiliazed, when needed,
            // since the kernel loop breaks at the first occurence of breaksize
        }

        return to_ret;
    }

}

namespace hd { // RNG

    /**
     * @brief Generate a random 0 or 1.
     *
     * @return the randomly generated 0 or 1
     */
    int random01();

    /**
     * @brief Generate a random -1 or 1.
     * 
     * @return int 
     */
    int random1sign();
}

namespace hd { //DEBUG FUNCTIONS


    /**
     * @brief Debug function, prints a matrix to the console
     * 
     * @tparam Tp Type of data in buffer
     * @param buffer 2d buffer to print out
     */
    template <typename Tp>
    void print_debug_2dbuffer(cl::sycl::buffer<Tp,2> &buffer){
        auto range = buffer.get_range();
        cl::sycl::host_accessor acc_buff(buffer, cl::sycl::read_only);
        for (int i = 0; i < range[0]; i++){
            for (int j = 0; j < range[1]; j++)
                std::cout << std::right << std::setw(6) << std::setprecision(2) <<acc_buff[i][j];
        std::cout << std::endl;
        }
    }

    template <typename Tp>
    void print_debug_2dbuffer(cl::sycl::buffer<Tp,2> &buffer, std::vector<std::string> &h_labels, std::vector<std::string> &v_labels, size_t w = 7){
        auto range = buffer.get_range();
        std::cout << std::right << std::setw(w) << std::setprecision(2) << "";
        cl::sycl::host_accessor acc_buff(buffer, cl::sycl::read_only);
        for (int i = 0; i < range[1]; i++)
            std::cout << std::right << std::setw(w) << h_labels[i];
        std::cout << std::endl;

        for (int i = 0; i < range[0]; i++){
            std::cout << std::right << std::setw(w) << v_labels[i];
            for (int j = 0; j < range[1]; j++)
                std::cout << std::right << std::setw(w) << acc_buff[i][j];
            std::cout << std::endl;
        }
    }

    template <typename Tp>
    void print_debug_1dbuffer(cl::sycl::buffer<Tp,1> &buffer){
        int range = buffer.get_range()[0];
        cl::sycl::host_accessor acc_buff(buffer, cl::sycl::read_only);
        for (int i = 0; i < range; i++)
            std::cout << acc_buff[i] << " ";
        std::cout << std::endl;
    }   

}

namespace hd { // index matrix to sycl_hd op type

    /**
     * @brief Matrix of indexes to sycl_hd operation type
     * 
     * @param idxs 
     * @return op_indexes 
     */
    op_indexes imat2op(std::vector<std::vector<uint>> &idxs);
    op_indexes imat2op(std::vector<std::vector<uint>> &idxs, uint t_size);

    /**
     * @brief Index 2d bbuffer to operation type
     * 
     * @param buffer 
     * @return op_indexes 
     */
    op_indexes bf2d2op(hd::index_2d_buffer &&buffer);

}



#endif //SYCLHD_UTILITIES_HPP