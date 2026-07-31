/**
 * @file rbf_enc.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief RBF encoding method implementations
 * @version 0.1
 * @date 2024-02-13
 * 
 * @copyright GPL3
 * 
 */

#include "rbf_enc.hpp"
#include <random>

namespace hd {


    /**
     * Generates a 2D buffer of float type using a normal distribution.
     *
     * @param Br 2D range for the buffer {dimensionality, number of features}
     * @param mean mean value for the normal distribution
     * @param std_dev standard deviation for the normal distribution
     *
     * @return 2D buffer of float vectors generated using the normal distribution
     */
    float_2d_buffer genBVectors(cl::sycl::range<2> Br, const float mean, const float std_dev, cl::sycl::queue *q) {

        float_2d_buffer B_vectors(Br);
        B_vectors.set_write_back(false);

        static std::random_device dev;
        static std::mt19937 rng(dev());
        static std::normal_distribution<float> gen{mean,std_dev};
        
        {        
            auto B_acc = B_vectors.get_host_access(cl::sycl::write_only, cl::sycl::no_init);
            for(int d = 0; d < Br[0]; d++){
                for(int f = 0; f < Br[1]; f++){
                    B_acc[d][f] = gen(rng);
                }
            }
        }

        if(q != nullptr)
            q->wait();

        return B_vectors;
    }

    /**
     * Encode feature vectors using the maniHD RBF method.
     * For some feature vector F the corresponding encoded vector H
     * is defined by
     *
     * h_i = cos(B_i . F)
     *
     * @param feature_vectors 2D buffer of feature vectors
     * @param B_vectors 2D buffer of B vectors
     * @param q SYCL queue for parallel processing
     *
     * @return encoded HVs
     */
    float_2d_buffer maniHDRBF(float_2d_buffer &feature_vectors, float_2d_buffer &B_vectors, cl::sycl::queue &q){

        const cl::sycl::range v_range(
            feature_vectors.get_range()[0], // Number of feature vectors
            B_vectors.get_range()[0]        // HV size
        );

        const size_t fts = B_vectors.get_range()[1]; // Number of features

        float_2d_buffer res(v_range);

        res.set_write_back(false);

        q.submit([&](cl::sycl::handler &h){
            auto fv_acc = feature_vectors.get_access(h,cl::sycl::read_only);
            auto B_acc = B_vectors.get_access(h,cl::sycl::read_only);
            auto res_acc = res.get_access(h,cl::sycl::write_only,cl::sycl::no_init);

            h.parallel_for(v_range,[=](cl::sycl::id<2> lr){

                auto fv_n = lr[0];
                auto hv_d = lr[1];

                // Dot product
                float v = 0.0f;
                for (int f = 0; f < fts; f++){
                    v += fv_acc[fv_n][f] * B_acc[hv_d][f];
                }

                res_acc[fv_n][hv_d] = cos(v);
            });
        });

        return res;
    }

    /**
     * [UNUSED] Encoded feature vectors using a modified NeuralHD RBF algorithm.
     * The modified is due to b value always being 0.
     *
     * @param feature_vectors The input feature vectors buffer
     * @param B_vectors The B vectors buffer
     * @param q The SYCL queue for execution
     *
     * @return encoded HVs
     */
    float_2d_buffer modneuralHDRBF(float_2d_buffer &feature_vectors, float_2d_buffer &B_vectors, cl::sycl::queue &q){

        const cl::sycl::range v_range(
            feature_vectors.get_range()[0], // Number of feature vectors
            B_vectors.get_range()[0]        // HV size
        );
        const size_t fts = B_vectors.get_range()[1]; // Number of features


        float_2d_buffer res(v_range);

        res.set_write_back(false);

        q.submit([&](cl::sycl::handler &h){
            auto fv_acc = feature_vectors.get_access(h,cl::sycl::read_only);
            auto B_acc = B_vectors.get_access(h,cl::sycl::read_only);
            auto res_acc = res.get_access(h,cl::sycl::write_only,cl::sycl::no_init);

            h.parallel_for(v_range,[=](cl::sycl::id<2> lr){


                const auto fv_n = lr[0];
                const auto hv_d = lr[1];

                // Dot product
                float v = 0;
                for (int k = 0; k < fts; k++){
                    v += fv_acc[lr[0]][k] * B_acc[lr[1]][k];
                }

                const float c = cl::sycl::cos(v);
                const float s = cl::sycl::sin(v);

                res_acc[fv_n][hv_d] = c*s;
            });
        });

        return res;
    }



    /**
     * Encode feature vectors usingthe NeuralHD RBF algorithm.
     *
     * For some feature vector F the corresponding encoded vector H
     * is defined by
     *
     * h_i = cos(B_i . F + b) * sin(B_i . F)
     *
     * @param feature_vectors the input 2D buffer of feature vectors
     * @param B_vectors the input 2D buffer of B vectors
     * @param b_values the input 1D buffer of b values
     * @param q the SYCL queue for the computation
     *
     * @return encoded HVs
     */
    float_2d_buffer neuralHDRBF(float_2d_buffer &feature_vectors, float_2d_buffer &B_vectors, float_1d_buffer &b_values, cl::sycl::queue &q){

        const cl::sycl::range<2> v_range(
            feature_vectors.get_range()[0], // Number of feature vectors
            B_vectors.get_range()[0]        // HV size
        );

        float_2d_buffer res(v_range);
        const size_t fts = B_vectors.get_range()[1]; // Number of features

        // Safety Checks
        // Might remove in the future since these are redundant
        // Commented out for now
        /* 
        if (fts != feature_vectors.get_range()[1]) 
            throw std::invalid_argument("Feature vector and B vector sizes must be equal");

        if (v_range[1] != b_values.get_range()[0])
            throw std::invalid_argument("b values need to match HDC problem dimensions"); 
        */

        q.submit([&](cl::sycl::handler &h){
            auto fv_acc = feature_vectors.get_access(h,cl::sycl::read_only);
            auto B_acc = B_vectors.get_access(h,cl::sycl::read_only);
            auto b_acc = b_values.get_access(h,cl::sycl::read_only);
            auto res_acc = res.get_access(h,cl::sycl::write_only,cl::sycl::no_init);

            h.parallel_for(v_range,[=](cl::sycl::id<2> lr){
                const auto fv_n = lr[0];
                const auto hv_d = lr[1];

                // Dot product
                float v = 0;
                for (int k = 0; k < fts; k++){
                    float r = fv_acc[fv_n][k] * B_acc[hv_d][k];
                    v += r;
                }

                // NeuralHD formula
                float b = b_acc[hv_d];
                float c = cl::sycl::cos(v + b);
                float s = cl::sycl::sin(v);

                res_acc[fv_n][hv_d] = c*s;
            });
        });

        return res;
    }

    /**
     * Encode feature vectors using the NeuralHD RBF algorithm, as above.
     * Automatically generates the b values and calls neuralHDRBF
     *
     * @param feature_vectors reference to the 2D buffer of feature vectors
     * @param B_vectors reference to the 2D buffer of B vectors
     * @param q reference to the SYCL queue
     *
     * @return a 2D buffer representing the neural HDRBF
     */
    float_2d_buffer neuralHDRBF(float_2d_buffer &feature_vectors, float_2d_buffer &B_vectors, cl::sycl::queue &q){

        static float_1d_buffer b_values(B_vectors.get_range()[0]);
        b_values.set_write_back(false);

        // Generate b values from range 0,2PI
        static auto gen_bv = [&](){
            static std::random_device dev;
            static std::mt19937 rng(dev());
            std::uniform_real_distribution<float> gen(0,2*M_PI);
            
            auto res_acc = b_values.get_host_access(cl::sycl::write_only,cl::sycl::no_init);
            for (auto &v : res_acc)
                v = gen(rng);
            return 0;
        };

        static int g = gen_bv();

        // If the b values buffer does not have the same number of rows as
        // the B vectors, regenerate b values. Otherwise, use the existing
        // b values buffer (which may have been modified in a previous run).
        g = b_values.get_range()[0] != B_vectors.get_range()[0];

        if (g){
            b_values = float_1d_buffer(B_vectors.get_range()[0]);
            g = gen_bv();
        }

        return neuralHDRBF(feature_vectors, B_vectors, b_values, q);
    }
}