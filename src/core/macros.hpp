/**
 * @file macros.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief MACROS
 * @version 0.2
 * @date 2024-02-21
 * 
 * @copyright GPL3
 * 
 */

#ifndef SYCLHD_MACROS_HPP
#define SYCLHD_MACROS_HPP

#define MODEL(TYPE)                              \
    template <typename Tp, typename buffer_type> \
    TYPE model<Tp,buffer_type,l_mat,ul_mat>

#define MATRIX(TYPE)                \
    template <typename buffer_type> \
    TYPE matrix<buffer_type>

#define L_MATRIX(TYPE)                                  \
    template <typename buffer_type, typename label_type>\
    TYPE l_matrix<buffer_type,label_type>

#define UL_MATRIX(TYPE)                                 \
    template <typename buffer_type, typename label_type>\
    TYPE ul_matrix<buffer_type,label_type>
    

#define RANDOM_DENSE_BIN(B_TYPE, KERNEL)                                                        \
    void randomDenseBinVectorGenerator(B_TYPE &buffer) {                                        \
        cl::sycl::range<2> buffer_range = buffer.get_range();                                   \
        cl::sycl::host_accessor acc_this_vectors(buffer,cl::sycl::write_only,cl::sycl::no_init);\
        for (auto &v_e: acc_this_vectors) KERNEL                                                \
    }


#define BASIC_ENCODER(NAME,TYPE,KERNEL) template<>                                                   \
matrix<TYPE> matrix<TYPE>::NAME(std::vector<index_1d_buffer> &indexes) {                              \
    size_t n_vecs = indexes.size();                                                                  \
    size_t v_size = this->hv_buffer.get_range()[1];                                                  \
    matrix<TYPE> to_ret(cl::sycl::range(n_vecs,v_size),this->q);                                     \
    for (uint e_i = 0; e_i < indexes.size(); e_i++){                                                 \
        size_t e_size = indexes[e_i].get_range()[0];                                                 \
        this->q.submit([&](cl::sycl::handler &h) {                                                   \
            cl::sycl::accessor acc_this(this->hv_buffer, h, cl::sycl::read_only);                    \
            cl::sycl::accessor acc_ret(to_ret.hv_buffer, h, cl::sycl::write_only, cl::sycl::no_init);\
            cl::sycl::accessor acc_data(indexes[e_i],h,cl::sycl::read_only);                         \
            h.parallel_for(v_size, [=](cl::sycl::id<1> dim) KERNEL);                                 \
        });                                                                                          \
    }                                                                                                \
    return to_ret;                                                                                   \
}

#define BASIC_ENCODER_OP(NAME,TYPE,KERNEL) template<>                               \
matrix<TYPE> matrix<TYPE>::NAME(op_indexes &indexes) {                              \
    size_t n_vecs = indexes.new_entry.get_range()[0]-1;                                  \
    size_t v_size = this->hv_buffer.get_range()[1];                                                  \
    cl::sycl::range<2> r(n_vecs,v_size);                                                              \
    matrix<TYPE> to_ret(cl::sycl::range(n_vecs,v_size),this->q);                                     \
    this->q.submit([&](cl::sycl::handler &h) {                                                   \
        cl::sycl::accessor acc_this(this->hv_buffer, h, cl::sycl::read_only);                    \
        cl::sycl::accessor acc_ret(to_ret.hv_buffer, h, cl::sycl::write_only, cl::sycl::no_init);\
        cl::sycl::accessor acc_data(indexes.indexes,h,cl::sycl::read_only);                         \
        cl::sycl::accessor acc_i(indexes.new_entry,h,cl::sycl::read_only);                          \
        h.parallel_for(r, [=](cl::sycl::id<2> dim) KERNEL);                                 \
    });                                                                                          \
    return to_ret;                                                                                   \
}

#define BASIC_ENCODER_OP_SHORT(NAME,TYPE,ACCUM_TYPE,OP,NORMALIZATION) template<>                               \
matrix<TYPE> matrix<TYPE>::NAME(op_indexes &indexes) {                              \
    size_t n_vecs = indexes.new_entry.get_range()[0]-1;                                  \
    size_t v_size = this->hv_buffer.get_range()[1];                                                  \
    cl::sycl::range<2> r(n_vecs,v_size);                                                              \
    matrix<TYPE> to_ret(r,this->q);                                     \
    this->q.submit([&](cl::sycl::handler &h) {                                                   \
        cl::sycl::accessor acc_this(this->hv_buffer, h, cl::sycl::read_only);                    \
        cl::sycl::accessor acc_ret(to_ret.hv_buffer, h, cl::sycl::write_only, cl::sycl::no_init);\
        cl::sycl::accessor acc_data(indexes.indexes,h,cl::sycl::read_only);                         \
        cl::sycl::accessor acc_i(indexes.new_entry,h,cl::sycl::read_only);                          \
        h.parallel_for(r, [=](cl::sycl::id<2> dim) {\
            ACCUM_TYPE v = acc_this[acc_data[acc_i[dim[0]]]][dim[1]];\
            for (int i = acc_i[dim[0]] + 1; i < acc_i[dim[0]+1]; i++)  \
                v OP acc_this[acc_data[i]][dim[1]];\
            acc_ret[dim] = NORMALIZATION; \
        });                                 \
    });                                                                                          \
    return to_ret;                                                                                   \
}

#define BASIC_ENCODER_SQUARE(NAME,TYPE,KERNEL) template<>                                        \
matrix<TYPE> matrix<TYPE>::NAME(index_2d_buffer &indexes) {                                         \
    size_t n_vecs = indexes.get_range()[0];                                                      \
    size_t v_size = this->hv_buffer.get_range()[1];                                              \
    matrix<TYPE> to_ret(cl::sycl::range(n_vecs,v_size),this->q);                                 \
    this->q.submit([&](cl::sycl::handler &h) {                                                   \
        cl::sycl::accessor acc_this(this->hv_buffer, h, cl::sycl::read_only);                    \
        cl::sycl::accessor acc_ret(to_ret.hv_buffer, h, cl::sycl::write_only, cl::sycl::no_init);\
        cl::sycl::accessor acc_data(indexes,h,cl::sycl::read_only);                              \
        h.parallel_for(to_ret.hv_buffer.get_range(), [=](cl::sycl::id<2> lr) KERNEL);            \
    });                                                                                          \
    return to_ret;                                                                               \
}

#define BIND_WITH(NAME,TYPE,OP) template<>                                                   \
matrix<TYPE> matrix<TYPE>::NAME(matrix<TYPE> &matrix_to_bind) {                                  \
    cl::sycl::range r(this->hv_buffer.get_range());                                              \
    matrix<TYPE> to_ret(r,this->q);                                                              \
    if (r != matrix_to_bind.hv_buffer.get_range()){                                              \
        if (matrix_to_bind.hv_buffer.get_range()[0] != 1)                                        \
            throw std::invalid_argument("Matrix vector size missmatch");                         \
        this->q.submit([&](cl::sycl::handler &h) {                                               \
            cl::sycl::accessor acc_ret(to_ret.hv_buffer, h, cl::sycl::write_only, cl::sycl::no_init);\
            cl::sycl::accessor acc_this(this->hv_buffer, h, cl::sycl::read_only);                    \
            cl::sycl::accessor acc_that(matrix_to_bind.hv_buffer, h, cl::sycl::read_only);           \
            h.parallel_for(r, [=](cl::sycl::id<2> i) {                                               \
                acc_ret[i] = acc_this[i] OP acc_that[0][i[1]];});                                 \
        });                                                                                      \
    } else                                                                                       \
    this->q.submit([&](cl::sycl::handler &h) {                                                   \
        cl::sycl::accessor acc_ret(to_ret.hv_buffer, h, cl::sycl::write_only, cl::sycl::no_init);\
        cl::sycl::accessor acc_this(this->hv_buffer, h, cl::sycl::read_only);                    \
        cl::sycl::accessor acc_that(matrix_to_bind.hv_buffer, h, cl::sycl::read_only);           \
        h.parallel_for(r, [=](cl::sycl::id<2> i) {acc_ret[i] = acc_this[i] OP acc_that[i];});    \
    });                                                                                          \
    return to_ret;                                                                               \
}

// BASE LEVEL

#define BASELEVEL_ENCODER(TYPE,KERNEL) template<>\
matrix<TYPE> matrix<TYPE>::baseLevel(matrix<TYPE> &Levels, index_2d_buffer &indexes) {\
    size_t n_base = indexes.get_range()[1];                                                          \
    if (n_base != this->hv_buffer.get_range()[0])                                                    \
        throw std::invalid_argument("Provided indexes must mach base matrix dimension");             \
    size_t n_vecs = indexes.get_range()[0];                                                          \
    size_t v_size = this->hv_buffer.get_range()[1];                                                  \
    matrix<TYPE> to_ret(cl::sycl::range(n_vecs,v_size),this->q);                                     \
    for (unsigned int j = 0; j < n_base; j++)                                                        \
        this->q.submit([&](cl::sycl::handler &h) {                                                   \
            cl::sycl::accessor acc_bases(this->hv_buffer, h, cl::sycl::read_only);                   \
            cl::sycl::accessor acc_levels(Levels.hv_buffer, h, cl::sycl::read_only);                 \
            cl::sycl::accessor acc_data(indexes, h, cl::sycl::read_only);                            \
            cl::sycl::accessor acc_ret(to_ret.hv_buffer, h, cl::sycl::write_only, cl::sycl::no_init);\
            h.parallel_for(to_ret.hv_buffer.get_range(), [=](cl::sycl::id<2> lr) KERNEL);            \
        });                                                                                          \
    return to_ret;                                                                                   \
}

#define BASELEVEL_ENCODER_INTERNAL_FOR(TYPE,KERNEL) template<>\
matrix<TYPE> matrix<TYPE>::baseLevel(matrix<TYPE> &Levels, index_2d_buffer &indexes) {\
    size_t n_base = indexes.get_range()[1]; \
    if (n_base != this->hv_buffer.get_range()[0])\
        throw std::invalid_argument("baseLevel: Provided indexes must mach base matrix dimension");\
    size_t n_vecs = indexes.get_range()[0];\
    size_t v_size = this->hv_buffer.get_range()[1];\
    matrix<TYPE> to_ret(cl::sycl::range(n_vecs,v_size),this->q);\
    this->q.submit([&](cl::sycl::handler &h) {\
        cl::sycl::accessor acc_bases(this->hv_buffer, h, cl::sycl::read_only);\
        cl::sycl::accessor acc_levels(Levels.hv_buffer, h, cl::sycl::read_only);\
        cl::sycl::accessor acc_data(indexes, h, cl::sycl::read_only);\
        cl::sycl::accessor acc_ret(to_ret.hv_buffer, h, cl::sycl::write_only, cl::sycl::no_init);\
        h.parallel_for(to_ret.hv_buffer.get_range(), [=](cl::sycl::id<2> lr) KERNEL);\
    });\
    return to_ret;\
}

#define BASELEVEL_ENCODER_NOT_SQUARE(B_TYPE,KERNEL)\
template <>\
matrix<B_TYPE>\
matrix<B_TYPE>::baseLevel(matrix<B_TYPE> &Levels,std::vector<index_1d_buffer> &indexes) {\
    size_t n_base = this->hv_buffer.get_range()[0];\
    size_t n_vecs = indexes.size();\
    size_t v_size = this->hv_buffer.get_range()[1];\
    std::vector<B_TYPE> rets(n_vecs,B_TYPE(cl::sycl::range(1,v_size)));\
    for (unsigned int vec = 0; vec < indexes.size(); vec++)\
        this->q.submit([&](cl ::sycl ::handler &h) {\
            cl::sycl::accessor acc_bases(this->hv_buffer, h, cl::sycl::read_only);\
            cl::sycl::accessor acc_levels(Levels.hv_buffer, h,cl::sycl::read_only);\
            cl::sycl::accessor acc_rets(rets[vec], h, cl::sycl::write_only, cl::sycl::no_init);\
            cl::sycl::accessor acc_data(indexes[vec], h, cl::sycl::read_only);\
            h.parallel_for(v_size,[=](cl::sycl::id<1> dim) KERNEL);\
        });\
    matrix<B_TYPE> to_ret(cl::sycl::range(n_vecs,v_size),this->q);\
    for (unsigned int vec = 0; vec < indexes.size(); vec++)\
        this->q.submit([&](cl ::sycl ::handler &h) {\
            cl::sycl::accessor acc_rets(rets[vec], h, cl::sycl::read_only);\
            cl::sycl::accessor acc_ret(to_ret.hv_buffer, h, cl::sycl::write_only, cl::sycl::no_init);\
            h.parallel_for(v_size,[=](cl::sycl::id<1> dim) {\
                acc_ret[vec][dim] = acc_rets[0][dim];\
            });\
        });\
    return to_ret;\
}
// NGRAM


/// ngram bundle kernel macro
#define NGRAM_BUNDLE_K(PERMUTATION,BUNDLE_TYPE,BIND_NEUTRAL,NORMALIZATION) \
{\
    size_t dt_point = lr[0];\
    size_t dim = lr[1];\
    size_t hv_i = acc_data[dt_point][0];\
    BUNDLE_TYPE v = acc_this[hv_i][dim];\
    for (auto dt_dim = 1; dt_dim < bundle_size; dt_dim++){\
        hv_i = acc_data[dt_point][dt_dim];\
        if (hv_i == this_size) break;\
        int pn = dt_dim % n_gram_size;\
        PERMUTATION\
        v += acc_this[hv_i][n_dim];\
    }\
    acc_ret[dt_point][dim] = NORMALIZATION;\
}

/// ngram bind kernel macro
#define NGRAM_BIND_K(PERMUTATION,BUNDLE_TYPE,BIND_NEUTRAL,NORMALIZATION) \
{ \
    size_t dt_point = lr[0];\
    size_t dim = lr[1];\
    size_t hv_i = acc_data[dt_point][0];\
    BUNDLE_TYPE v(0);\
    for (size_t dt_dim_M = 0; dt_dim_M < bundle_size/n_gram_size; dt_dim_M++){\
        size_t dt_dim = dt_dim_M*n_gram_size;\
        BIND_NEUTRAL;\
        for (auto pn = 0; (pn < n_gram_size && dt_dim_M*n_gram_size + pn < bundle_size); pn++){\
            hv_i = acc_data[dt_point][dt_dim];\
            if (hv_i == this_size) break;\
            PERMUTATION\
            c = c * acc_this[hv_i][n_dim];\
            dt_dim += 1;\
        }\
        if (hv_i == this_size) break;\
        v += c;\
    }\
    acc_ret[dt_point][dim] = NORMALIZATION;\
}



#define NGRAM_SUBMIT(KERNEL)\
    q.submit([&](cl::sycl::handler &h) { \
        cl::sycl::accessor acc_this(this->hv_buffer, h, cl::sycl::read_only); \
        cl::sycl::accessor acc_data(indexes,h,cl::sycl::read_only); \
        cl::sycl::accessor acc_ret(to_ret.hv_buffer, h, cl::sycl::write_only, cl::sycl::no_init); \
        h.parallel_for(to_ret_r, [=](cl::sycl::id<2> lr) KERNEL);\
    });\

#define NGRAM(NAME,TYPE,KERNEL,BUNDLE_TYPE,BIND_NEUTRAL,NORMALIZATION)\
template<> matrix<TYPE> matrix<TYPE>::NAME(index_2d_buffer &indexes, const size_t n_gram_size, const int shift_depth){\
    size_t n_vecs = indexes.get_range()[0];                         \
    size_t vs = this->hv_buffer.get_range()[1];                     \
    matrix<TYPE> to_ret(cl::sycl::range(n_vecs,vs),this->q);        \
    cl::sycl::range<2> to_ret_r(to_ret.hv_buffer.get_range());      \
    size_t bundle_size = indexes.get_range()[1];                    \
    size_t this_size = this->hv_buffer.get_range()[0];              \
    if (shift_depth == 0) throw std::invalid_argument(              \
        "shift_depth cannot be 0"                                   \
    );                                                              \
    else if (shift_depth == 1) NGRAM_SUBMIT(KERNEL(                 \
        size_t n_dim = (dim + pn) % vs;                             \
    ,BUNDLE_TYPE,BIND_NEUTRAL,NORMALIZATION))                        \
    else if (shift_depth == -1) NGRAM_SUBMIT(KERNEL(                \
        size_t n_dim = (vs + dim - pn) % vs;                        \
    ,BUNDLE_TYPE,BIND_NEUTRAL,NORMALIZATION))                        \
    else NGRAM_SUBMIT(KERNEL(                                       \
        size_t n_dim = (vs + dim + (pn*shift_depth)) % vs;          \
    ,BUNDLE_TYPE,BIND_NEUTRAL,NORMALIZATION))                        \
    return to_ret;                                                  \
}

#define NGRAM_BIND_BUNDLE(TYPE,BUNDLE_TYPE,BIND_NEUTRAL,NORMALIZATION)\
    NGRAM(ngramBind, TYPE, NGRAM_BIND_K,BUNDLE_TYPE,BIND_NEUTRAL,NORMALIZATION)\
    NGRAM(ngramBundle, TYPE, NGRAM_BUNDLE_K,BUNDLE_TYPE,BIND_NEUTRAL,NORMALIZATION)\


#define NGRAM_SUBMIT_BL(KERNEL)\
    q.submit([&](cl::sycl::handler &h) { \
        cl::sycl::accessor acc_this(this->hv_buffer, h, cl::sycl::read_only); \
        cl::sycl::accessor acc_data(indexes,h,cl::sycl::read_only); \
        cl::sycl::accessor acc_levels(Levels.hv_buffer,h,cl::sycl::read_only); \
        cl::sycl::accessor acc_ret(to_ret.hv_buffer, h, cl::sycl::write_only, cl::sycl::no_init); \
        h.parallel_for(to_ret_r, [=](cl::sycl::id<2> lr) KERNEL);\
    });\

#define NGRAM_BL_K(PERMUTATION,BUNDLE_TYPE,NORMALIZATION)\
{\
    size_t dt_point = lr[0];\
    size_t dim = lr[1];\
    size_t dt_dim = 0;\
    size_t hv_i = acc_data[dt_point][dt_dim];\
    auto v_base = acc_this[dt_dim][dim];\
    auto v_lvl = acc_levels[hv_i][dim];\
    BUNDLE_TYPE v = v_base*v_lvl;\
    for (dt_dim = 1; dt_dim < bundle_size; dt_dim++){\
        hv_i = acc_data[dt_point][dt_dim];\
        if (hv_i == lvl_size) break;\
        int pn = dt_dim % n_gram_size;\
        PERMUTATION\
        v_base = acc_this[dt_dim][n_dim];\
        v_lvl = acc_levels[hv_i][n_dim];\
        v += v_base*v_lvl;\
    }\
    acc_ret[dt_point][dim] = NORMALIZATION;\
}

#define NGRAM_BL(TYPE,BUNDLE_TYPE,NORMALIZATION)\
template<> matrix<TYPE> matrix<TYPE>::ngramBaseLevel(matrix &Levels, index_2d_buffer &indexes, const size_t n_gram_size, const int shift_depth){\
    size_t n_vecs = indexes.get_range()[0];                         \
    size_t vs = this->hv_buffer.get_range()[1];                     \
    matrix<TYPE> to_ret(cl::sycl::range(n_vecs,vs),this->q);        \
    cl::sycl::range<2> to_ret_r(to_ret.hv_buffer.get_range());      \
    size_t bundle_size = indexes.get_range()[1];                    \
    size_t this_size = this->hv_buffer.get_range()[0];              \
    size_t lvl_size = Levels.hv_buffer.get_range()[0];              \
    if (shift_depth == 0) throw std::invalid_argument(              \
        "shift_depth cannot be 0"                                   \
    );                                                              \
    if (bundle_size != this_size) throw std::invalid_argument(      \
        "index size missmatch with bases: " + std::to_string(bundle_size) + " != " + std::to_string(this_size)     \
        );                                                              \
    else if (shift_depth == 1) NGRAM_SUBMIT_BL(NGRAM_BL_K(          \
        size_t n_dim = (dim + pn) % vs;                             \
    ,BUNDLE_TYPE,NORMALIZATION))                                                              \
    else if (shift_depth == -1) NGRAM_SUBMIT_BL(NGRAM_BL_K(         \
        size_t n_dim = (vs + dim - pn) % vs;                        \
    ,BUNDLE_TYPE,NORMALIZATION))                                                              \
    else NGRAM_SUBMIT_BL(NGRAM_BL_K(                                \
        size_t n_dim = (vs + dim + (pn*shift_depth)) % vs;          \
    ,BUNDLE_TYPE,NORMALIZATION))                                                              \
    return to_ret;                                                  \
}


// QUERY
#define COSINE_DISTANCE_MATRIX(TYPE, KERNEL_L)\
float_2d_buffer cosineDistanceMatrix(TYPE &buff_keys, TYPE &buff_test, cl::sycl::queue &q) {\
    cl::sycl::range<2> v_range(\
        buff_test.get_range()[0],\
        buff_keys.get_range()[0]\
    );\
    float_2d_buffer buff_distance_vectors(v_range);\
    size_t vs = buff_keys.get_range()[1];\
    q.submit([&](cl::sycl::handler &h) {\
        cl::sycl::accessor acc_dist_vectors(buff_distance_vectors, h, cl::sycl::write_only, cl::sycl::no_init);\
        cl::sycl::accessor acc_encoded_test_entries(buff_test, h,cl::sycl::read_only);\
        cl::sycl::accessor acc_model_entries(buff_keys, h, cl::sycl::read_only);\
        h.parallel_for(v_range, [=](cl::sycl::id<2> local_range) {\
            size_t i = local_range[0];\
            size_t j = local_range[1];\
            float num=0,den1=0,den2=0;\
            for (size_t k = 0; k < vs; k++) KERNEL_L\
            float den = cl::sycl::sqrt(den1*den2);\
            acc_dist_vectors[i][j] = num/den;\
        });\
    });\
    return buff_distance_vectors;\
}


#endif //SYCLHD_MACROS_HPP