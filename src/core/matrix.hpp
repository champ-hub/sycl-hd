/**
 * @file matrix.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief HDC Matrix object
 * @version 0.2
 * @date 2024-01-26
 *
 * TODO: should matrix inherit from sycl buffer 2d?
 *       remove method bodies from class declarations (l/ul matrix)
 *       add visual and doc seperation from methods defined in modules and defined here
 *
 * @copyright GPL3
 * 
 */

#ifndef HDC_MATRIX_HPP
#define HDC_MATRIX_HPP

#include "utilities.hpp"
#include "query.hpp"
#include "training.hpp"


namespace hd{ // MATRIX BASE CLASS

    template <typename buffer_type, typename label_type>
    class l_matrix;
    template <typename buffer_type, typename label_type>
    class ul_matrix;

    /**
     * @brief HDC Matrix Parent Class
     *
     * Contains a SYCL buffer of templated type
     * 
     * @tparam buffer_type must be a 2D SYCL buffer: cl::sycl::buffer<Tp,2>
     */
    template <typename buffer_type, int id = -1>
    class matrix{

        public:
        
            // === MEMBERS ===
            buffer_type hv_buffer; ///< SYCL buffer that contains the matrix
            cl::sycl::queue &q; ///< Associated SYCL queue

            // === CONSTRUCTORS AND INIT ===

            matrix(buffer_type &&buffer, cl::sycl::queue &queue) : hv_buffer(buffer), q(queue){}

            matrix(matrix<buffer_type> &mat): hv_buffer(mat.hv_buffer.get_range()), q(mat.q)
            {
                this->q.submit([&](cl::sycl::handler &h){
                    auto this_acc = this->hv_buffer.get_access(h,cl::sycl::write_only, cl::sycl::no_init);
                    auto that_acc = mat.hv_buffer.get_access(h,cl::sycl::read_only);
                    h.parallel_for(this->hv_buffer.get_range(),[=](cl::sycl::id<2> i){
                        this_acc[i[0]][i[1]] = that_acc[i[0]][i[1]];
                    });
                });
                this->hv_buffer.set_write_back(false);
            }

            matrix(cl::sycl::range<2> range, cl::sycl::queue &queue):
                q(queue), hv_buffer(range)
            {
                this->hv_buffer.set_write_back(false);
            }

            matrix(cl::sycl::range<2> range, cl::sycl::queue &queue, vectors_generator gen) : 
                matrix(range,queue) {generateVectors(gen);
            }

            matrix(size_t vs, size_t n_vectors, cl::sycl::queue &queue) : 
                q(queue), hv_buffer(cl::sycl::range(n_vectors,vs)){
                this->hv_buffer.set_write_back(false);
            }

            matrix(size_t vs, size_t n_vectors, cl::sycl::queue &queue, vectors_generator gen) : 
                matrix(vs,n_vectors,queue) {generateVectors(gen);
            }

            matrix(matrix<buffer_type> &a, matrix<buffer_type> &b) : 
                q(a.q), 
                hv_buffer(cl::sycl::range(
                    a.hv_buffer.get_range()[0] + b.hv_buffer.get_range()[0],
                    a.hv_buffer.get_range()[1]
                ))
            {

                if (a.hv_buffer.get_range()[1] != b.hv_buffer.get_range()[1])
                    throw std::invalid_argument("Matrices to stack must match dimensions");

                size_t sizea = a.hv_buffer.get_range()[0];

                this->q.submit([&](cl::sycl::handler &h) {
                    cl::sycl::accessor acc_a(a.hv_buffer, h, cl::sycl::read_only);
                    cl::sycl::accessor acc_b(b.hv_buffer, h, cl::sycl::read_only);
                    cl::sycl::accessor acc_this(this->hv_buffer, h, cl::sycl::write_only, cl::sycl::no_init);
                    h.parallel_for(this->hv_buffer.get_range(), [=](cl::sycl::id<2> i) {
                        if (i[0] < sizea)
                            acc_this[i] = acc_a[i];
                        else
                            acc_this[i[0]][i[1]] = acc_b[i[0]-sizea][i[1]];
                    });
                });
            }

            matrix(std::vector<matrix<buffer_type>*> &&mats) : 
                q(mats[0]->q), 
                hv_buffer(cl::sycl::range(
                    [&](){
                        size_t sz = 0;
                        for(auto mat: mats)
                            sz += mat->hv_buffer.get_range()[0];
                        return sz;
                    }(),
                    mats[0]->hv_buffer.get_range()[1]
                ))
            {
                size_t offset = 0;
                for(auto &mat: mats){
                    if (mat->hv_buffer.get_range()[1] != this->hv_buffer.get_range()[1])
                        throw std::invalid_argument("Matrices to stack must match dimensions");

                    this->q.submit([&](cl::sycl::handler &h) {
                        cl::sycl::accessor acc_a(mat->hv_buffer, h, cl::sycl::read_only);
                        cl::sycl::accessor acc_this(this->hv_buffer, h, cl::sycl::write_only, cl::sycl::no_init);
                        h.parallel_for(mat->hv_buffer.get_range(), [=](cl::sycl::id<2> i) {
                            acc_this[i[0]+offset][i[1]] = acc_a[i];
                        });
                    });
                    offset += mat->hv_buffer.get_range()[0];
                }
                this->hv_buffer.set_write_back(false);
            }
            /**
             * @brief Generate Vectors for this matrix
             * 
             * @param gen vector generator selector defined in "definitions.hpp"
             */
            void generateVectors(vectors_generator gen);

            /**
             * @brief Set the HV buffer of this matrix from a vector of vectors
             * 
             * @param vecs2set Vector of Vectors, with compatible type with this matrix
             */
            void setVectors(std::vector<std::vector<typename buffer_type::value_type>> &vecs2set);


            [[nodiscard]] size_t getVectorSize();

            /**
             * @brief Get the Vectors from this matrix HV buffer
             * 
             * @return std::vector<std::vector<typename buffer_type::value_type>> 
             */
            [[nodiscard]] std::vector<std::vector<typename buffer_type::value_type>> getVectors();

            [[nodiscard]] matrix bind(std::vector<index_1d_buffer> &indexes);
            [[nodiscard]] matrix bind(op_indexes &indexes);
            [[nodiscard]] matrix bundle(std::vector<index_1d_buffer> &indexes);
            [[nodiscard]] matrix bundle(op_indexes &indexes);

            /**
             * @brief Bundle the entries of this matrix by the given labels
             * Also known as one shot training in the context of HDC
             * 
             * @tparam lt label type
             * @param labels vector of repeating labels to bundle to each other, 
             *      has the same size as number of HVs in this matrix
             * @param show_fit if true, will print the training fit
             * @return ul_matrix<buffer_type, lt> unique labled matrix with the trained (bundled) class HVs
             */
            template<typename lt>
            [[nodiscard]] ul_matrix<buffer_type, lt> bundleByLabels(std::vector<lt> const& labels, const bool show_fit = false);

            /**
             * @brief Trains the HDC classification model
             * Calls the bundleByLabels method if no retraining is required
             * Else, the _train method handles the retraining
             * 
             * @tparam lt label type
             * @param labels vector of labels to train on, has same size as number of HVs in this matrix
             * @param rt_steps Retraining iterations
             * @param alpha Alpha value for OnlineHD and AdaptiveHD retraining
             * @param method Retraining method selector as defined in "definitions.hpp"
             * @param show_fit if true, will print the training fit
             * @return ul_matrix<buffer_type, lt> unique labled matrix with the trained (bundled) class HVs
             */
            template<typename lt> 
            [[nodiscard]] ul_matrix<buffer_type, lt> train(std::vector<lt> const& labels, const int rt_steps = 0, const float alpha = -1.0f, retrain method = adapthd, const bool show_fit = false);
            

            [[nodiscard]] matrix invert();
            [[nodiscard]] matrix unbind(matrix<buffer_type> &matrix_to_bind);
            
            /**
             * @brief Bind with another identically sized matrix
             * 
             * @param matrix_to_bind 
             * @return matrix 
             */
            [[nodiscard]] matrix bindWith(matrix<buffer_type> &matrix_to_bind);

            /**
             * @brief Base Level encoder method
             * 
             * @param Levels 
             * @param indexes 
             * @return matrix 
             */
            [[nodiscard]] matrix baseLevel(matrix<buffer_type> &Levels, index_2d_buffer &indexes);
            [[nodiscard]] matrix baseLevel(matrix<buffer_type> &Levels, std::vector<index_1d_buffer> &indexes);

            template<typename label_type>
            matrix<buffer_type> baseLevel(ul_matrix<buffer_type,label_type> &Levels, std::vector<std::vector<label_type>> &labels_to_enc){
                index_2d_buffer indexes = Levels.getRectLabelsInIdxRep(labels_to_enc);
                return this->baseLevel(Levels, indexes);
            }

            template<typename label_type>
            matrix<buffer_type> baseLevelNotSquareData(ul_matrix<buffer_type,label_type> &Levels, std::vector<std::vector<label_type>> &labels_to_enc){
                std::vector<index_1d_buffer> indexes = Levels.getLabelsInIdxRep(labels_to_enc);
                return this->baseLevel(Levels, indexes);
            }

            template<typename Tp>
            [[nodiscard]] matrix ngram(std::vector<std::vector<Tp>> const& indexes, const size_t n_gram_sz, const permutation p = shift_left, const bool bind = false, const bool barrel_shift = false){
                if (barrel_shift) {
                    auto n_indexes = barrelshift(indexes, n_gram_sz, this->hv_buffer.get_range()[0]);
                    return this->ngram(n_indexes,n_gram_sz,p,bind);
                } else {
                    auto n_indexes = mat2RectIdxBuff(indexes, this->hv_buffer.get_range()[0]);
                    return this->ngram(n_indexes,n_gram_sz,p,bind);
                }
            }

            [[nodiscard]] matrix ngram(index_2d_buffer &indexes, const size_t n_gram_sz = 3, permutation p = shift_left, const bool bind = false){
                if (n_gram_sz > this->hv_buffer.get_range()[1])
                    throw std::invalid_argument("Ngram size excedes this HV dimesionality or buffer size");
                if (bind) return this->ngramBind(indexes, n_gram_sz, p);
                return this->ngramBundle(indexes, n_gram_sz, p);
            }

            /**
             * @brief Ngram Base Level
             * Called from the base matrix object
             * ngram = (Base0*Level_i1) + p(Base1*Level_i2) + pp(Base2*Level_i3) + ...
             * encoded = ngram_1 + ngram_2 + ...
             * 
             * @param indexes 
             * @param n_gram_sz 
             * @param shift_depth 
             * @return matrix 
             */
            [[nodiscard]] matrix ngramBaseLevel(matrix &Levels, index_2d_buffer &indexes, const size_t n_gram_sz, const int shift_depth = 1);

            template<typename Tp>
            [[nodiscard]] matrix ngramBaseLevel(matrix &Levels, std::vector<std::vector<Tp>> const& indexes, const size_t n_gram_sz, const int shift_depth = 1){
                auto idx = mat2RectIdxBuff(indexes,Levels.hv_buffer.get_range()[0]);
                return ngramBaseLevel(Levels,idx,n_gram_sz,shift_depth);
            }

            [[nodiscard]] matrix bindDown();
            [[nodiscard]] matrix bundleDown();

            /**
             * @brief Stack a matrix on top of this one 
             * 
             * Does not modify this object
             *
             * @param to_stack 
             * @return matrix<buffer_type> 
             */
            [[nodiscard]] matrix stack(matrix<buffer_type> &to_stack){
                return matrix(*this,to_stack);
            }

            [[nodiscard]] matrix operator[] (const size_t s);

            /**
             * @brief Get indexes of highest similarity with entries
             * 
             * @param to_query entries to query
             * @return index_1d_buffer 
             */
            [[nodiscard]] index_1d_buffer queryIdx(matrix<buffer_type> &to_query);

            /**
             * @brief Return similarity Matrix between the HVs of this matrix (T)
             * and the HVs of matrix and `to_query` (Q)
             * 
             * sim(T1,Q1), sim(T2,Q1), ... , sim(Tn,Q1)
             * sim(T1,Q2), sim(T2,Q2), ... , sim(T2,Q2)
             * ...
             * sim(T1,Qm), sim(T2,Qm), ... , sim(Tn,Qm)
             *
             * 
             * @param to_query 
             * @return float_2d_buffer 
             */
            [[nodiscard]] float_2d_buffer distanceMatrix(matrix<buffer_type> &to_query);
            
            /**
             * @brief Get the Variance for each dimension
             * 
             * @param normalized If the variance should be outputed according to 
             * the correct definition (with the last division step), set to false when called by some other methods
             * to save performance, since it is not needed for the comparison of variances to do the division step
             * @return float_1d_buffer value of the variace for each dimension of the matrix
             */
            [[nodiscard]] float_1d_buffer getVarianceVector(const bool normalized = true);
            
            /**
             * @brief Get the Buffer Acessor object
             * 
             * @return cl::sycl::host_accessor
             */
            [[nodiscard]] cl::sycl::host_accessor<typename buffer_type::value_type,2,cl::sycl::access_mode::read_write> getBufferAcessor(){
                return this->hv_buffer.get_host_access(cl::sycl::read_write);
            }

            /**
             * @brief Outpus a copy of this matrix but with float data
             * 
             * @return matrix<float_2d_buffer> 
             */
            [[nodiscard]] matrix<float_2d_buffer> toFloatMatrix();

            /**
             * @brief Copy the contents of a matrix to this one
             * 
             * @param input 
             */
            void cpyFrom(matrix<float_2d_buffer> &input);

            /**
             * @brief Randomly regenerate the specified dimensions
             * 
             * @param dims_to_regen 
             */
            void regenerate(index_1d_buffer &dims_to_regen);
            
            /**
             * @brief Get the accuracy of this model by comparing the
             * obtained indexes of highest similarity with the correct ones
             * provided by the user
             * 
             * @param to_querry entries to query
             * @param idx_answers corresponding true indexes
             * @return double 
             */
            double test(matrix<buffer_type> &to_querry, index_1d_buffer &idx_answers);

            /**
             * @brief Normalize each HV of this matrix, only available for MAP
             * 
             */
            void normalize();

        protected:
        
            /**
             * @brief This function handles the retraining
             *
             * Retraining is done by one of three options: VoiceHD, AdaptHD or OnlineHD
             * Selected by the method parameter.
             *
             * VoiceHD retraining: On miss classification add HV to the correct class
             * and subract it to the wrong class
             *
             * AdaptHD retraining: On miss classification add HV*alpha to the correct class
             * and subract it to the wrong class
             *
             * OnlineHD retraining: On miss classification add HV*sim(HV,class)*alpha 
             * to the correct class and subract this expression to the wrong class
             *
             * @param occurences 
             * @param class_idx 
             * @param rt_steps retraining steps (breaks loop at stagnation of misses)
             * @param alpha alpha value for OnlineHD (if negative use VoiceHD)
             *
             * @return the retrained matrix
             */
            [[nodiscard]] matrix _train(op_indexes &occurences, index_1d_buffer &class_idx, const int rt_steps, const float alpha, const retrain method);
            
            /**
             * @brief Ngram bind
             * Called from ngram
             * ngram = HV * p(HV) * pp(HV2) * ...
             * encoded = ngram_1 + ngram_2 + ...
             * 
             * @param indexes 
             * @param n_gram_sz 
             * @param shift_depth 
             * @return matrix 
             */
            [[nodiscard]] matrix ngramBind(index_2d_buffer &indexes, const size_t n_gram_sz, const int shift_depth);

            /**
             * @brief Ngram Bundle
             * Called from ngram
             * ngram = HV + p(HV) + pp(HV2) + ...
             * encoded = ngram_1 + ngram_2 + ...
             *
             * @param indexes 
             * @param n_gram_sz 
             * @param shift_depth 
             * @return matrix 
             */
            [[nodiscard]] matrix ngramBundle(index_2d_buffer &indexes, const size_t n_gram_sz, const int shift_depth);
    };
}

namespace hd{ // LABELED MATRIX

    /**
     * @brief HDC Labled Matrix
     *
     * Inherits from `matrix` contains a vector of labels which 
     * may or may not be of unique values. The vector type is templated
     * 
     * @tparam buffer_type 
     * @tparam label_type 
     */
    template <typename buffer_type, typename label_type>
    class l_matrix : public matrix<buffer_type>{

    public:
        std::vector<label_type> labels;

        l_matrix(matrix<buffer_type> mat, std::vector<label_type> &ls):
            matrix<buffer_type>(mat), labels(ls)
        {}

        l_matrix(size_t vs, std::vector<label_type> &labels_to_set, cl::sycl::queue &queue, vectors_generator gen = none) : 
            matrix<buffer_type>(vs,labels_to_set.size(),queue,gen), labels(labels_to_set)
        {}

        l_matrix(l_matrix<buffer_type,label_type> &a, l_matrix<buffer_type,label_type> &b) : 
            matrix<buffer_type>(a,b)
        {
            this->labels.reserve(a.labels.size() + b.labels.size());
            this->labels.insert(this->labels.end(), a.labels.begin(), a.labels.end());
            this->labels.insert(this->labels.end(), b.labels.begin(), b.labels.end());
        }

        std::vector<label_type> query(matrix<buffer_type> &to_querry);

        std::vector<label_type> getLabels(){return this->labels;}
        std::vector<label_type> getLabels(index_1d_buffer &labels_to_get){

            std::vector<label_type> to_ret(labels_to_get.get_range()[0]);
            auto acc_idx = labels_to_get.get_host_access(cl::sycl::read_only);

            #pragma omp parallel for 
            for (int i = 0; i < to_ret.size(); i++){
                to_ret[i] = this->labels[acc_idx[i]];
            }

            return to_ret;
        }

        void setLabels(std::vector<label_type> const& labels){
            this->labels = labels;
        }

    };
}

namespace hd { // UNIQUE LABELS MATRIX

    /**
     * @brief Unique Labels HDC Matrix
     * 
     * Inherits from labeled matrix `l_matrix` contains an unordered map 
     * of label - index correspondence, must have unique labels.
     *
     * @tparam buffer_type 
     * @tparam label_type 
     */
    template <typename buffer_type, typename label_type>
    class ul_matrix : public l_matrix<buffer_type,label_type>{
    
    protected:
        
        void populateMap(){
            for (ushort i = 0; i < this->labels.size(); i++)
                this->label_idx_map.insert(std::pair(this->labels[i], i));
        }

    public:

        // MEMBERS
        std::unordered_map<label_type, ushort> label_idx_map;

        // CONSTRUCTORS
        ul_matrix(size_t vs, std::vector<label_type> &labels_to_set, cl::sycl::queue &queue, vectors_generator gen = none) : 
            l_matrix<buffer_type,label_type>(vs,labels_to_set,queue,gen), label_idx_map(labels_to_set.size())
        {populateMap();}

        ul_matrix(matrix<buffer_type> &&mat, std::vector<label_type> &ls):
            l_matrix<buffer_type,label_type>(mat,ls)
        {populateMap();}

        ul_matrix(l_matrix<buffer_type,label_type> &l_mat_cpy) : 
            l_matrix<buffer_type, label_type>(&l_mat_cpy) , label_idx_map(l_mat_cpy.labels.size())
        {populateMap();}

        ul_matrix(ul_matrix<buffer_type,label_type> &a, ul_matrix<buffer_type,label_type> &b) : 
            l_matrix<buffer_type,label_type>(a,b), label_idx_map(a.label_idx_map.size()+b.label_idx_map.size())
        {populateMap();}


        // SPECIFIC METHODS
        std::vector<index_1d_buffer> getLabelsInIdxRep(std::vector<std::vector<label_type>> &labels_to_find);
        index_2d_buffer getRectLabelsInIdxRep(std::vector<std::vector<label_type>> &labels_to_find);

        using matrix<buffer_type>::baseLevel;
        using matrix<buffer_type>::baseLevelNotSquareData;
        using matrix<buffer_type>::ngram;

        matrix<buffer_type> ngram(std::vector<std::vector<label_type>> &labels_to_enc, const size_t n_gram_sz = 3, permutation p = shift_left, const bool bind = false){
            index_2d_buffer indexes = this->getRectLabelsInIdxRep(labels_to_enc);
            return this->ngram(indexes, n_gram_sz,p,bind);
        }


        using matrix<buffer_type>::operator[];

        [[nodiscard]] matrix<buffer_type> operator[] (const label_type s){
            return this->operator[](this->label_idx_map[s]);
        }
        
        ul_matrix<buffer_type, label_type> stack(ul_matrix<buffer_type, label_type> &to_stack){
            return ul_matrix<buffer_type, label_type>(*this, to_stack);
        }
        
        void setLabels(std::vector<label_type> const& labels);

        using matrix<buffer_type>::test;

        double test(matrix<buffer_type> &to_querry, std::vector<label_type> const& label_answers);

    };
}

namespace hd { // MATRIX HEADER DEFINED FUNCTIONS

    template<typename buffer_type, int id>
    void matrix<buffer_type, id>::setVectors(std::vector<std::vector<typename buffer_type::value_type>> &vecs2set){
        cl::sycl::range<2> r(vecs2set.size(), vecs2set[0].size());
        if (r != this->hv_buffer.get_range()) throw std::invalid_argument("Dimension missmatch setVectors");

        auto acc = this->hv_buffer.get_host_access(cl::sycl::write_only, cl::sycl::no_init);

        for(int i = 0; i < r[0]; i++)
            for(int j = 0; j < r[1]; j++)
                acc[i][j] = vecs2set[i][j];
    }

    template<typename buffer_type, int id>
    std::vector<std::vector<typename buffer_type::value_type>> matrix<buffer_type,id>::getVectors(){
        auto acc = this->hv_buffer.get_host_access(cl::sycl::read_only);
        std::vector<std::vector<typename buffer_type::value_type>> to_ret(
            acc.get_range()[0],
            std::vector<typename buffer_type::value_type>(acc.get_range()[1])
        );

        for(auto i = 0; i < acc.get_range()[0]; i++)
            for(auto j = 0; j < acc.get_range()[1]; j++)
                to_ret[i][j] = acc[i][j];
            
        return to_ret;
    }

    template<typename buffer_type, int id>
    template<typename lt>
    ul_matrix<buffer_type, lt> matrix<buffer_type,id>::bundleByLabels(std::vector<lt> const& labels, const bool show_fit) {
        p_labels processed_labels = _processReduceLabels(labels);
        ul_matrix<buffer_type, lt> to_ret(this->bundle(processed_labels.op_indexes),processed_labels.unique_labels);
        if (show_fit){ 
            auto tf = to_ret.test(*this, labels)*100;
            std::cout <<
                "train fit: "<< tf <<"%" << std::endl;
        }
        return to_ret;
    }

    template<typename buffer_type, int id>
    matrix<buffer_type,id> matrix<buffer_type,id>::_train(
        op_indexes &occurences, 
        index_1d_buffer &class_idx, 
        const int rt_steps, 
        const float alpha,
        const retrain method
    ) {
        
        matrix<float_2d_buffer> float_vecs = this->toFloatMatrix();
        matrix<float_2d_buffer> accumulator = float_vecs.bundle(occurences);
        cl::sycl::range<2> r(occurences.new_entry.get_range()[0]-1, this->hv_buffer.get_range()[1]);

        matrix<buffer_type> to_ret(r,this->q);

        int l_misses = 0; // Misses in last iteration
        int stagnation = 0; // Stagnation counter

        for (int rt_step = 1; rt_step <= rt_steps; rt_step++){
            auto distance_matrix = accumulator.distanceMatrix(float_vecs);
            int misses = 0;
            switch (method){
                case retrain::voicehd:
                    misses = retrainIterationVoiceHD(accumulator.hv_buffer, float_vecs.hv_buffer, distance_matrix, class_idx, this->q);
                    break;
                case retrain::onlinehd:
                    misses = retrainIterationOnlineHD(accumulator.hv_buffer, float_vecs.hv_buffer, distance_matrix, class_idx, this->q,alpha);
                    break;
                case retrain::adapthd:
                    misses = retrainIterationAdaptHD(accumulator.hv_buffer, float_vecs.hv_buffer, distance_matrix, class_idx, this->q,alpha);
                    break;
            }
            
            //std::cout << "Iteration: " << rt_step << std::endl;
            //std::cout << "Misses: " << misses << std::endl;
            
            if (misses == l_misses) stagnation++;
            else stagnation = 0;

            if (misses == 0 || stagnation > 3) break;

            l_misses = misses;
        }
        to_ret.cpyFrom(accumulator);
        return to_ret;
    }
    

    template<typename buffer_type, int id>
    template<typename lt>
    ul_matrix<buffer_type, lt> matrix<buffer_type, id>::train(std::vector<lt> const& labels, const int rt_steps, const float alpha, retrain method, const bool show_fit) {
        if (rt_steps <= 0) return this->bundleByLabels(labels, show_fit);
        if (alpha <= 0) method = voicehd;

        p_labels bundle_idxs = _processReduceLabels(labels);

        ul_matrix<buffer_type, lt> to_ret(
            this->_train(
                bundle_idxs.op_indexes,
                bundle_idxs.label_nidx_corr,
                rt_steps,alpha,method
            ),
            bundle_idxs.unique_labels
        );

        if (show_fit){ 
            auto tf = to_ret.test(*this, bundle_idxs.label_nidx_corr)*100;
            std::cout <<
                "train fit: "<< tf <<"%" << std::endl;
        }

        return to_ret;

    }

    /**
     * @brief Return Hypervector at index
     *
     * Returns a matrix of 1 vector copied from the specified index from this matrix
     * 
     * @tparam buffer_type 
     * @param s 
     * @return matrix<buffer_type> 
     */
    template <typename buffer_type, int id>
    matrix<buffer_type, id> matrix<buffer_type, id>::operator[](const size_t s) {
        matrix<buffer_type, id> to_ret(this->hv_buffer.get_range()[1], 1, this->q);
        this->q.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc_ret(to_ret.hv_buffer, h, cl::sycl::write_only, cl::sycl::no_init);
            cl::sycl::accessor acc_this(this->hv_buffer, h, cl::sycl::read_only);
            h.parallel_for(to_ret.hv_buffer.get_range(), [=](cl::sycl::id<2> i) {
                acc_ret[i] = acc_this[s][i[1]];
            });
        });
        return to_ret;
    };

    /**
     * @brief Get indexes of highest similarity with entries
     * 
     * @tparam buffer_type 
     * @param to_query 
     * @return index1d_buffer 
     */
    template <typename buffer_type, int id>
    index_1d_buffer
    matrix<buffer_type, id>::queryIdx(matrix<buffer_type> &to_query) {
        float_2d_buffer sim_mat = this->distanceMatrix(to_query);
        return idxOfHighestValueInMatRow(sim_mat, this->q);
    }

    /**
     * @brief Bind all rows of this matrix
     * 
     * @tparam buffer_type 
     * @return matrix<buffer_type> 
     */
    template <typename buffer_type, int id>
    matrix<buffer_type, id> matrix<buffer_type, id>::bindDown() {
      index_1d_buffer sequential_indexes(this->hv_buffer.get_range()[0]);
      this->q.submit(
          [&](cl::sycl::handler &h) {
            cl::sycl::accessor acc_ret(sequential_indexes, h, cl::sycl::write_only, cl::sycl::no_init);
            h.parallel_for(sequential_indexes.get_range(), [=](cl::sycl::id<1> i) {
                acc_ret[i] = i;
            });
          });
      std::vector<index_1d_buffer> idxs{sequential_indexes};
      return bind(idxs);
    };

    /**
     * @brief Bundle all rows of this matrix
     * 
     * @tparam buffer_type 
     * @return matrix<buffer_type> 
     */
    template <typename buffer_type, int id>
    matrix<buffer_type, id> matrix<buffer_type, id>::bundleDown() {
      index_1d_buffer sequential_indexes(this->hv_buffer.get_range()[0]);
      this->q.submit(
          [&](cl::sycl::handler &h) {
            cl::sycl::accessor acc_ret(sequential_indexes, h, cl::sycl::write_only, cl::sycl::no_init);
            h.parallel_for(sequential_indexes.get_range(), [=](cl::sycl::id<1> i) {
                acc_ret[i] = i;
            });
          });
      std::vector<index_1d_buffer> idxs = std::vector{sequential_indexes};
      return bundle(idxs);
    };

    /**
     * @brief Regenerates the specified dimensions for each HV of this matrix
     * TODO: maybe shuffling would be faster than regeneration of random numbers
     *
     * @param dims_to_regen the index of dimensions to regenerate
     */
    template <typename buffer_type, int id>
    void matrix<buffer_type,id>::regenerate(index_1d_buffer &dims_to_regen){
        cl::sycl::range<2> regen_range(this->hv_buffer.get_range()[0],dims_to_regen.get_range()[0]);

        matrix<buffer_type> regenerated(regen_range, this->q, random);

        this->q.submit([&](cl::sycl::handler &h){
            auto this_acc = this->hv_buffer.get_access(h,cl::sycl::write_only, cl::sycl::no_init);
            auto that_acc = regenerated.hv_buffer.get_access(h,cl::sycl::read_only);
            auto dims_acc = dims_to_regen.get_access(h,cl::sycl::read_only);
            h.parallel_for(regen_range,[=](cl::sycl::id<2> i){
                auto this_dim = dims_acc[i[1]];
                this_acc[i[0]][this_dim] = that_acc[i[0]][i[1]];
            });
        });
    }

    template<typename buffer_type, int id>
    double matrix<buffer_type, id>::test(matrix<buffer_type> &to_querry, index_1d_buffer& idx_answers){
        index_1d_buffer idx = this->queryIdx(to_querry);
        index_1d_buffer ret(1);
        const size_t idxr = idx_answers.get_range()[0];
        this->q.submit([&](cl::sycl::handler &h){
           auto ans = idx_answers.get_access(h,cl::sycl::read_only);
           auto idx_acc = idx.get_access(h,cl::sycl::read_only); 
           auto ret_acc = ret.get_access(h,cl::sycl::write_only,cl::sycl::no_init);
           h.single_task([=](){
               int ret = 0;
               for (int i = 0; i < idxr; i++){
                   if (ans[i] == idx_acc[i]){
                       ret++;
                   }
               }
               ret_acc[0] = ret;
           });
        });

        return (double) ret.get_host_access(cl::sycl::read_only)[0] / idx_answers.get_range()[0];

    }

}

namespace hd { // LABELED MATRIX HEADER DEFINED FUNCTIONS

    /**
     * @brief Query Matrix return closest vector labels
     * 
     * @tparam buffer_type 
     * @tparam label_type 
     * @param to_querry 
     * @return std 
     */
    template <typename buffer_type, typename label_type>
    std ::vector<label_type>
    l_matrix<buffer_type, label_type>::query(matrix<buffer_type> &to_querry) {
        index_1d_buffer idx = this->queryIdx(to_querry);
        return this->getLabels(idx);
    };
}

namespace hd { // UNIQUE LABELS MATRIX HEADER DEFINED FUNCTIONS

    /**
     * @brief Get Labels in Index Representation
     *
     * From a vector of vectors of labels returs their corresponding index inside this matrix
     * as a vector of 1D unsigned integer buffers
     * 
     * @tparam buffer_type 
     * @tparam label_type 
     * @param labels_to_find 
     * @return std 
     */
    template <typename buffer_type, typename label_type>
    inline std ::vector<index_1d_buffer>
    ul_matrix<buffer_type, label_type>::getLabelsInIdxRep(
        std::vector<std::vector<label_type>> &labels_to_find
    ) {
        std::vector<index_1d_buffer> to_ret;
        to_ret.reserve(labels_to_find.size());
        for (auto vec : labels_to_find) {

            index_1d_buffer buff_to_ins(vec.size());
            cl::sycl::host_accessor acc_to_ins(buff_to_ins, cl::sycl::write_only,cl::sycl::no_init);
            
            for (int i = 0; i < vec.size(); i++) 
                acc_to_ins[i] = this->label_idx_map[vec[i]];
    
            to_ret.emplace_back(buff_to_ins);
        }
        return to_ret;
    }

    /**
     * @brief Get Labels in Index Representation
     *
     * Same function as homonym but assumes the provided vector of vectors of labels
     * has all elements of same size. Returns a 2D unsigned integer buffer of corresponding
     * index from this matrix 
     * 
     * @tparam buffer_type 
     * @tparam label_type 
     * @param labels_to_find 
     * @return index_2d_buffer 
     */
    template <typename buffer_type, typename label_type>
    inline index_2d_buffer
    ul_matrix<buffer_type, label_type>::getRectLabelsInIdxRep(
        std::vector<std::vector<label_type>> &labels_to_find
    ) {
        index_2d_buffer to_ret(cl::sycl::range(labels_to_find.size(), labels_to_find[0].size()));
        cl::sycl::host_accessor acc_to_ret(to_ret, cl::sycl::write_only,cl::sycl::no_init);
        for (int i = 0; i < labels_to_find.size(); i++) 
            for (int j = 0; j < labels_to_find[i].size(); j++) 
                acc_to_ret[i][j] = this->label_idx_map[labels_to_find[i][j]];
        
        return to_ret;
    }

    /**
     * @brief Get Accuracy of this AM
     * 
     * @tparam buffer_type 
     * @tparam label_type 
     * @param to_querry Entries to querry
     * @param label_answers Answers
     * @return double Accuracy (0 to 1)
     */
    template <typename buffer_type, typename label_type>
    inline double
    ul_matrix<buffer_type, label_type>::test(
        matrix<buffer_type> &to_querry, std::vector<label_type> const& label_answers
    ) {
        if (to_querry.hv_buffer.get_range()[0] != label_answers.size())
            throw std::invalid_argument("Input dimension missmatch: labels and query");
        std::vector<label_type> test_labels = this->query(to_querry);
        int ans = 0;
        for (uint i = 0; i < test_labels.size(); i++)
            ans += (label_answers[i] == test_labels[i]);

        SYCL_HD_PRINT("test(): Hits: " + std::to_string(ans) + " out of " + std::to_string(test_labels.size()));
        
        return (double)ans / test_labels.size();
    }

    /**
     * @brief Set Labels of this Unique Labels Matrix
     * 
     * @tparam buffer_type 
     * @tparam label_type 
     * @param labels Unique labels for each HV
     */
    template <typename buffer_type, typename label_type>
    inline void
    ul_matrix<buffer_type, label_type>::setLabels(std::vector<label_type> const& labels){
        this->labels = labels;
        this->label_idx_map = std::unordered_map<label_type, ushort>(this->labels.size());
        for (ushort i = 0; i < this->labels.size(); i++)
            this->label_idx_map.insert(std::pair(this->labels[i], i));
    }


}

#endif //HDC_MATRIX_HPP