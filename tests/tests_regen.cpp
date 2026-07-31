/**
 * @file tests_regen.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief NeuralHD Regeneration Tests
 * @version 0.1
 * @date 2024-03-09
 * 
 * @copyright GPL3
 * 
 */

#include "tests.hpp"
#include <syclhd.hpp>
#ifndef DPHDC_TESTS_HPP
#error Tests not defined!
#endif

#define V_SIZE 16
#define N_VECS 1000
#define MARGIN 0.02f

#define VAR_BERN_PROB 0.25f // Expected variance for Bernoulli
#define VAR_BIPO_PROB 1.00f // Expected variance for i.i.d. Bernoulli from set (-1,1)

/**
 * Test for variance calculation using map Model
 * WARNING: Might fail due to randomness but it is very unlikely
 */
TEST(variance,map){

    hd::model model = hd::MAP(V_SIZE);

    auto var_r = 
        model
            .genUnlabeledAtomicRep(N_VECS)
            .getVarianceVector()
            .get_host_access(cl::sycl::read_only);

    auto var_0 = 
        model
            .genUnlabeledAtomicRep(N_VECS, hd::all_true)
            .getVarianceVector()
            .get_host_access(cl::sycl::read_only);

    for (int i = 0; i < V_SIZE; i++){
        EXPECT_NEAR(var_r[i],VAR_BIPO_PROB,MARGIN);
        EXPECT_NEAR(var_0[i],0,MARGIN);
    }
}

TEST(variance,mapbp){

    hd::model model = hd::TMAP(V_SIZE);

    auto var_r = 
        model
            .genUnlabeledAtomicRep(N_VECS)
            .getVarianceVector()
            .get_host_access(cl::sycl::read_only);

    auto var_0 = 
        model
            .genUnlabeledAtomicRep(N_VECS, hd::all_true)
            .getVarianceVector()
            .get_host_access(cl::sycl::read_only);

    for (int i = 0; i < V_SIZE; i++){
        EXPECT_NEAR(var_r[i],VAR_BIPO_PROB,MARGIN);
        EXPECT_NEAR(var_0[i],0,MARGIN);
    }
}

TEST(variance,bsc){

    auto model = hd::BSC(V_SIZE);

    auto var_r = 
        model
            .genUnlabeledAtomicRep(N_VECS)
            .getVarianceVector()
            .get_host_access(cl::sycl::read_only);

    auto var_0 = 
        model
            .genUnlabeledAtomicRep(N_VECS, hd::all_true)
            .getVarianceVector()
            .get_host_access(cl::sycl::read_only);

    for (int i = 0; i < V_SIZE; i++){
        EXPECT_NEAR(var_r[i],VAR_BERN_PROB,MARGIN);
        EXPECT_NEAR(var_0[i],0,MARGIN);
    }
}

TEST(selectIndexes,selectIndexes){

    const float ratio = 0.2f;
    const float vsize = 10; 
    
    std::vector<float> v(vsize,0.0f);
    std::iota(v.begin(), v.end(), 2.1f);

    hd::float_1d_buffer var_vec(v.begin(),v.end());

    auto idxs = hd::selectIndexes(var_vec,ratio);
    auto acc = idxs.get_host_access(cl::sycl::read_only);

    std::vector<size_t> ans(ratio*vsize);
    std::iota(ans.begin(), ans.end(), 0);
    
    for (int i = 0; i<acc.size(); i++) {
        EXPECT_EQ(ans[i], acc[i]);
    }
}

TEST(regen,range){
    auto model = hd::MAP(V_SIZE);
    auto feature_gen = hd::MAP(24);

    auto feature_vectors = feature_gen.genUnlabeledAtomicRep(10).hv_buffer;

    auto AM = model.RBFKernelTrick(feature_vectors);

    cl::sycl::range expected(10,V_SIZE);
    EXPECT_EQ(AM.hv_buffer.get_range(), expected);
}

// TEST of a private method, confirmed working
/* TEST(regen, regenBmat_row){

    hd::model model = hd::BSC(V_SIZE);
    model.B_vectors = hd::genBVectors(cl::sycl::range<2>(10,10), 0.0f,0.0f);
    
    //hd::print_debug_2dbuffer(model.B_vectors);
    //std::cout << std::endl;
    //std::cout << std::endl;

    hd::index_1d_buffer rows2regen(1);
    rows2regen.get_host_access(cl::sycl::write_only, cl::sycl::no_init)[0] = 0;

    model._regenerateBmatRows(rows2regen);
    //hd::print_debug_2dbuffer(model.B_vectors);

    auto B_acc = model.B_vectors.get_host_access(cl::sycl::read_only);
    for (int i = 0; i < 10; i++){
        EXPECT_NE(B_acc[0][i], 0.0f);
    }

} */

TEST(regen, regenBmat){

    auto model = hd::BSC(V_SIZE);
    
    model.B_vectors = hd::genBVectors(cl::sycl::range<2>(10,10), 0.0f,0.0f);
    
    auto feature_vectors = hd::MAP(V_SIZE).genUnlabeledAtomicRep(10).hv_buffer;
    auto AM = model.RBFKernelTrick(feature_vectors);

    //std::cout << "AM Before: " << std::endl;
    //hd::print_debug_2dbuffer(AM.hv_buffer);

    model.regenerate(AM,{},1.0f);

    auto AM2 = model.RBFKernelTrick(feature_vectors);

    //std::cout << "AM After: " << std::endl;
    //hd::print_debug_2dbuffer(AM2.hv_buffer);

    auto B_acc = model.B_vectors.get_host_access(cl::sycl::read_only);
    for (int i = 0; i < V_SIZE; i++){
        EXPECT_NE(B_acc[i][i], 0.0f);
    }

    EXPECT_NE(AM.getVectors(),AM2.getVectors());
}