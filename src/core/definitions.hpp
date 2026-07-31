/**
 * @file definitions.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief Selectors (enums) and type definitions for the library
 * @version 2.0
 * @date 2024-01-31
 *
 * @copyright GPL3
 * 
 */
#ifndef HDC_DEFS_HPP
#define HDC_DEFS_HPP

#include <CL/sycl.hpp>


// Debug macros
#ifdef SYCL_HD_DEBUG
#define SYCL_HD_PRINT(MESSAGE)\
    std::cout << "SYCL_HD_DEBUG: " << MESSAGE << std::endl;
#else
#define SYCL_HD_PRINT(MESSAGE)
#endif



/// Main namespace for the library
namespace hd {

    /// Selector to ignore warning
    enum warning{
        show,  ///< Default for functions that produce warnings
        ignore ///< Option to disable warnings
    };

    /**
     * @brief Supported permutations
     *
     * It is possible to cast and integer to permutation type
     * like so: 
     *
     * hd::permutation p = (hd::permutation) 8
     *
     * resulting in a permutation depth of 8 elements to the left
     */
    enum permutation {
        shift_right = -1, ///< Circular shift right
        shift_left = 1    ///< Circular shift left
    };
    
    /// Suported Atomic Hypervector generators
    enum vectors_generator {
        none,       ///< Do not generate (buffer data is uninitialized)
        all_zero,   ///< Set all components to 0
        all_false,  ///< Set all components to false or equivalent
        all_true,   ///< Set all components to true or equivalent
        random,     ///< Set components randomly to true or false
        half_level, ///< Half level generation
        full_level, ///< Full level generation
        circular    ///< Circular level generation
    };

    /// Supported RBF methods
    enum rbf_method {
        cos_dist,      ///< Cosine distance
        cos_dot,       ///< Cosine of dot product (ManiHD method)
        default_method ///< Default NeuralHD method
    };

    /// Supported retraining methods
    enum retrain{
        voicehd, ///< VoiceHD retraining
        adapthd, ///< AdaptHD retraining
        onlinehd ///< OnlineHD retraining
    };

    // Buffer type used for storing indexes
    using index_1d_buffer = cl::sycl::buffer<unsigned short, 1>;
    using index_2d_buffer = cl::sycl::buffer<unsigned short, 2>;

    // Buffer type used for storing signed integers
    using int_1d_buffer = cl::sycl::buffer<int, 1>;
    using int_2d_buffer = cl::sycl::buffer<int, 2>;

    // Buffer type used for storing real number values (floats)
    using float_1d_buffer = cl::sycl::buffer<float, 1>;
    using float_2d_buffer = cl::sycl::buffer<float, 2>;

    // Operation indexes type accepted by some functions
    // The first field 'indexes' stores the indexes of each entry sequentially
    // The 'new_entry' stores the index where a new entry starts
    // For example:
    // Bind indexes 1 4 7
    // Bind indexes 2 5 8 10
    // Bind indexes 3 6 9
    // Then:
    // indexes = 1 4 7 2 5 8 10 3 6 9
    // new_entry = 0 3 7 10
    // Usefull for 'non square' operations where each operation might have a different number of indexes
    struct op_indexes {
        index_1d_buffer indexes;
        index_1d_buffer new_entry;
    };

}

#endif //HDC_DEFS_HPP
