/**
 * @file problem.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief Model base class header
 * @version 0.1
 * @date 2024-01-24
 * 
 * @copyright GPL3
 * 
 */

#ifndef HDC_PROBLEM_HPP
#define HDC_PROBLEM_HPP

#include "matrix.hpp"
#include "rbf_enc.hpp" 

namespace hd { // PROBLEM PARENT CLASS
    
    /**
     * @brief Generic class for storing model problem information
     *
     * The user should use the specialized classes implemented in the same header file
     * This class (and derived) are useful for correctly generating HDC matrix objetcs
     * that are compatible within the same problem.
     *
     * TODO: Name this model or problem?
     * 
     * @tparam Tp 
     * @tparam buffer_type 
     */
    template<typename Tp, int id = -1>
    class model{
    protected:

        using buffer_type = cl::sycl::buffer<Tp,2>;

        const int model_id = -1;
        const std::string associated_model_type;
        const ushort mul;
        size_t vector_size;

        cl::sycl::queue associated_queue;


        void adjustFreeDims(warning = show);


        //Sets Gets
        //std::string getModelName();
        //std::string getDeviceName();

        matrix<float_2d_buffer> _RBFKernelTrickFloatMat(float_2d_buffer &feature_vectors_buff, const float mean, const float std_dev, const rbf_method method);

        matrix<float_2d_buffer> _RBFKernelTrickFloatMat(std::vector<std::vector<float>> &feature_vectors, const float mean, const float std_dev, const rbf_method method){
            float_2d_buffer feature_vectors_buff = mat2RectBuff(feature_vectors);
            return this->_RBFKernelTrickFloatMat(feature_vectors_buff,mean,std_dev,method);
        }

        void _regenerate(float_1d_buffer &&var_vec, std::vector<matrix<buffer_type>*> &atoms, const float ratio);
        void _regenerateBmatRows(index_1d_buffer &rows2regen);



    public:
        model(size_t vs = 10000, std::string m = "Undefined", ushort multiplier = 1, warning w = show) : 
            model(vs,m,multiplier,cl::sycl::queue{},w)
        {}

        model(size_t vs, std::string m, ushort multiplier, cl::sycl::queue queue, warning w = show) : 
            vector_size(vs), associated_model_type(m), mul(multiplier), associated_queue(queue), B_vectors(cl::sycl::range(0,0))
        {
            adjustFreeDims(w);
        }
        
        // Generate Atoms

        matrix<buffer_type> genUnlabeledAtomicRep(size_t n_vectors, vectors_generator gen = random);

        /**
         * @brief Generate Small (< 5 recommended) Labeled Atomic Representation
         * 
         * @tparam label_type 
         * @param labels_to_set 
         * @param gen 
         * @return l_matrix<buffer_type, label_type> 
         */
        template<typename label_type>
        l_matrix<buffer_type, label_type> genSmallAtomicRep(std::vector<label_type> &labels_to_set, vectors_generator gen = random, warning warn = show){
            throw std::runtime_error("genSmallAtomicRep not fully implemented");
            if (labels_to_set.size() >= 5 && warn == show) 
                std::cout << "Consider using `genAtomicRep()` for large representations (>= 5 vectors)" << std::endl
                          << "To hide this warning set the last parameter of this function to hdc::ignore" << std::endl;
            return l_matrix<buffer_type,label_type>(this->vector_size/this->mul, labels_to_set, this->associated_queue, gen);
        };

        /**
         * @brief Generate Atomic Representation for unique labels 
         * 
         * @tparam label_type 
         * @param labels_to_set 
         * @param gen 
         * @return ul_matrix<buffer_type, label_type> 
         */
        template<typename label_type>
        ul_matrix<buffer_type, label_type> genAtomicRep(std::vector<label_type> &labels_to_set, vectors_generator gen = random, warning warn = show){
            //if (labels_to_set.size() < 5 && warn == show) 
            //    std::cout << "Consider using `genSmallAtomicRep()` for small representations (< 5 vectors)" << std::endl
            //              << "To hide this warning set the last parameter of this function to hd::ignore" << std::endl;
            return ul_matrix<buffer_type,label_type>(this->vector_size/this->mul, labels_to_set, this->associated_queue, gen);
        };

        /**
         * B_vectors used for RBF kernel trick encoding
         * 2D float matrix where for B[i][j]:
         * i -> HV dimension n
         * j -> feature n
         * 
         */
        float_2d_buffer B_vectors;

        /**
         * @brief Get the Vector Size
         * 
         * @return size_t 
         */
        size_t getVectorSize() {return this->vector_size;};


        /**
         * @brief Get reference to associated queue
         * 
         * @return cl::sycl::queue& 
         */
        cl::sycl::queue &getQueue(){return this->associated_queue;}

        void setQueue(const cl::sycl::queue q){this->associated_queue = q;}

        /**
         * @brief Get the Associated Device Name
         * 
         * @return const std::string 
         */
        const std::string getDeviceName(){return this->associated_queue.get_device().template get_info<cl::sycl::info::device::name>();}

        /**
         * @brief Get the Model Name
         * 
         * @return const std::string& 
         */
        const std::string &getModelName(){return this->associated_model_type;}


        void regenerate(matrix<buffer_type> &AM, std::vector<matrix<buffer_type>*> atoms = {}, const float ratio = 0.1){
            return this->_regenerate(AM.getVarianceVector(false),atoms,ratio);
        }

        template<typename lt>
        void regenerate(ul_matrix<buffer_type,lt> &AM, std::vector<matrix<buffer_type>*> atoms = {}, const float ratio = 0.1){
            return this->_regenerate(AM.getVarianceVector(false),atoms,ratio);
        }

        matrix<buffer_type> RBFKernelTrick(float_2d_buffer &feature_vectors_buff, const float mean = 0, const float std_dev = 1, const rbf_method method = default_method);
        matrix<buffer_type> RBFKernelTrick(std::vector<std::vector<float>> &feature_vectors_mat, const float mean = 0, const float std_dev = 1, const rbf_method method = default_method);


    };

}

namespace hd { // HEADER DEFINE FUNCTIONS

    /**
     * @brief Adjust to Free Dimensions
     *
     * Calculates and Adjust problem to available free dimensions that may appear
     * when using bit-packed data types
     * 
     */
    template <typename Tp, int id>
    inline void model<Tp, id>::adjustFreeDims(warning w) {
        int remainder = (this->vector_size % this->mul);
        if (remainder > 0) {
            int free_dims = this->mul - remainder;
            this->vector_size += free_dims;
            if (w == show){
                std::cout << "Using vector size of " << vector_size
                            << " (takes advantage of " << free_dims << " free dimensions)"
                            << std::endl;
                std::cout << "To disable this message set you problem vector size to "
                            << vector_size << std::endl;
            }
        }
    };

    // Generate Atoms

    /**
     * @brief Generate Unlabeled Atomic Representation
     * 
     * @param n_vectors 
     * @param gen 
     * @return matrix<buffer_type> 
     */
    template <typename Tp, int id>
    inline matrix<typename model<Tp,id>::buffer_type>
    model<Tp,id>::genUnlabeledAtomicRep(size_t n_vectors,vectors_generator gen) {
        return matrix<buffer_type,id>(this->vector_size / this->mul, n_vectors, this->associated_queue, gen);
    };

    template <typename Tp, int id>
    inline void model<Tp,id>::_regenerate(float_1d_buffer &&var_vec, std::vector<matrix<buffer_type>*> &atoms, const float ratio){

        index_1d_buffer idxs2regen_buff = selectIndexes(var_vec, ratio);
        idxs2regen_buff.set_write_back(false);

        for (matrix<buffer_type>* atom : atoms) {
            atom->regenerate(idxs2regen_buff);
        }

        if (this->B_vectors.get_range()[0] != 1) 
            _regenerateBmatRows(idxs2regen_buff);
    }

    template <typename Tp, int id>
    inline void model<Tp, id>::_regenerateBmatRows(index_1d_buffer &rows2regen){

        cl::sycl::range regen_range(rows2regen.get_range()[0],this->B_vectors.get_range()[1]);
        auto regenerated = genBVectors(regen_range, 0.0, 1.0,&this->associated_queue);

        this->associated_queue.submit([&](cl::sycl::handler &h){
            auto this_acc = this->B_vectors.get_access(h,cl::sycl::write_only);
            auto that_acc = regenerated.get_access(h,cl::sycl::read_only);
            auto rows_acc = rows2regen.get_access(h,cl::sycl::read_only);
            h.parallel_for(regen_range,[=](cl::sycl::id<2> i){
                auto this_row = rows_acc[i[0]];
                this_acc[this_row][i[1]] = that_acc[i[0]][i[1]];
            });
        });
    }

    template <typename Tp, int id>
    inline matrix<float_2d_buffer> model<Tp,id>::_RBFKernelTrickFloatMat(float_2d_buffer &feature_vectors_buff, const float mean, const float std_dev, const rbf_method method){
        
        cl::sycl::range Br(this->vector_size, feature_vectors_buff.get_range()[1]);
        if (Br != this->B_vectors.get_range()){
            SYCL_HD_PRINT("_RBFKernelTrickFloatMat(): Generating B matrix with range " << std::to_string(Br[0]) << "x" << std::to_string(Br[1]) << "" << " (missmatch of preexisting B matrix)"); 
            this->B_vectors = genBVectors(Br,mean,std_dev,&this->associated_queue);
        }
        
        switch (method) {
            case cos_dist:
                return matrix<float_2d_buffer>(cosineDistanceMatrix(this->B_vectors, feature_vectors_buff, this->associated_queue),this->associated_queue);
            case cos_dot:
                return matrix<float_2d_buffer>(maniHDRBF(feature_vectors_buff, this->B_vectors, this->associated_queue),this->associated_queue);
            default:
                return matrix<float_2d_buffer>(neuralHDRBF(feature_vectors_buff, this->B_vectors, this->associated_queue),this->associated_queue);
        }
    }


    template <typename Tp, int id>
    inline matrix<typename model<Tp,id>::buffer_type> model<Tp, id>::RBFKernelTrick(float_2d_buffer &feature_vectors_buff, const float mean, const float std_dev, const rbf_method method){

        cl::sycl::range r(feature_vectors_buff.get_range()[0], this->vector_size/this->mul);
        matrix<buffer_type> to_ret(r,this->associated_queue);

        matrix<float_2d_buffer> float_mat = _RBFKernelTrickFloatMat(feature_vectors_buff,mean,std_dev,method);

        to_ret.cpyFrom(float_mat);
        return to_ret;
    }

    template <typename Tp, int id>
    inline matrix<typename model<Tp,id>::buffer_type> model<Tp,id>::RBFKernelTrick(std::vector<std::vector<float>> &feature_vectors_mat, const float mean, const float std_dev, const rbf_method method){
        cl::sycl::range r(feature_vectors_mat.size(), this->vector_size/this->mul);
        matrix<buffer_type,id> to_ret(r,this->associated_queue);

        matrix<float_2d_buffer> float_mat = _RBFKernelTrickFloatMat(feature_vectors_mat,mean,std_dev,method);
        
        to_ret.cpyFrom(float_mat);
        return to_ret;
    }

}


#endif //HDC_PROBLEM_HPP 