/**
 * @file tests_general.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-13
 * 
 * @copyright GPL3
 * 
 */

#include "tests.hpp"
#include <syclhd.hpp>
#ifndef DPHDC_TESTS_HPP
#error Tests not defined!
#endif

#define VEC_SIZE 8*20 //ENSURE DIVISIBLE BY 8 for test to pass

static hd::model tmap_prob = hd::TMAP(VEC_SIZE-1);
static hd::model bsc_prob = hd::BSC(VEC_SIZE-1);
static hd::model map_prob = hd::MAP(VEC_SIZE);

TEST(problem, use_free_dims){

    auto mapbp_mat = tmap_prob.genUnlabeledAtomicRep(1,hd::none);
    auto bsc_mat = bsc_prob.genUnlabeledAtomicRep(1,hd::none);

    EXPECT_EQ(mapbp_mat.getVectorSize(), VEC_SIZE);
    EXPECT_EQ(bsc_mat.getVectorSize(), VEC_SIZE);
}


TEST(vec_gen,level_bsc){
    auto bsc_mat = bsc_prob.genUnlabeledAtomicRep(10,hd::full_level);
    auto buff_sim = bsc_mat.distanceMatrix(bsc_mat);

    /* 
    std::cout.precision(3);
    std::cout << "SIMILARITY MATRIX BSC LEVEL" << std::endl;
    hd::print_debug_2dbuffer(buff_sim);
    std::cout<<std::endl;
    */    

    auto acc = buff_sim.get_host_access(cl::sycl::read_only);

    for (int i = 0; i < 10; i++) 
        for (int j = 0; j < 10; j++)
            EXPECT_FLOAT_EQ(acc[i][j], acc[j][i]);

}



TEST(query,sim_l_mat){

    std::vector<std::string> l = {"a","b","c","d"};

    auto map_mat = map_prob.genAtomicRep(l);
    auto mapbp_mat = tmap_prob.genAtomicRep(l);
    auto bsc_mat = bsc_prob.genAtomicRep(l);

    auto buff_sim_map = map_mat.distanceMatrix(map_mat);
    auto buff_sim_bsc = bsc_mat.distanceMatrix(bsc_mat);
    auto buff_sim_mapbp = mapbp_mat.distanceMatrix(mapbp_mat);

    /* 
    std::cout.precision(4);

    std::cout << "SIMILARITY MATRIX MAP BP" << std::endl;
    hd::print_debug_2dbuffer(buff_sim_mapbp);
    std::cout<<std::endl;

    std::cout << "SIMILARITY MATRIX MAP" << std::endl;
    hd::print_debug_2dbuffer(buff_sim_map);
    std::cout<<std::endl;

    std::cout << "SIMILARITY MATRIX BSC" << std::endl;
    hd::print_debug_2dbuffer(buff_sim_bsc);
    std::cout<<std::endl; 
    */

    auto acc_map_v = buff_sim_map.get_host_access(cl::sycl::read_only);
    auto acc_bsc_v = buff_sim_bsc.get_host_access(cl::sycl::read_only);
    auto acc_mapbp_v = buff_sim_mapbp.get_host_access(cl::sycl::read_only);

    for (int i = 0; i<l.size() ; i++) {
            EXPECT_EQ(1, acc_map_v[i][i]);
            EXPECT_EQ(1, acc_bsc_v[i][i]);
            EXPECT_EQ(1, acc_mapbp_v[i][i]);
    }

    auto map_v = map_mat.query(map_mat);
    auto bsc_v = bsc_mat.query(bsc_mat);
    auto mapbp_v = mapbp_mat.query(mapbp_mat);

    EXPECT_EQ(map_v, l);
    EXPECT_EQ(bsc_v, l);
    EXPECT_EQ(mapbp_v, l);

}


TEST(query,same_mat){

    std::array<int, 5> ans = {0,1,2,3,4}; 

    auto map_mat = map_prob.genUnlabeledAtomicRep(5);
    auto mapbp_mat = tmap_prob.genUnlabeledAtomicRep(5);
    auto bsc_mat = bsc_prob.genUnlabeledAtomicRep(5);
    
    auto buff_map_v = map_mat.queryIdx(map_mat);
    auto buff_bsc_v = bsc_mat.queryIdx(bsc_mat);
    auto buff_mapbp_v = mapbp_mat.queryIdx(mapbp_mat);

    auto acc_map_v = buff_map_v.get_host_access(cl::sycl::read_only);
    auto acc_bsc_v = buff_bsc_v.get_host_access(cl::sycl::read_only);
    auto acc_mapbp_v = buff_mapbp_v.get_host_access(cl::sycl::read_only);

    std::array<int, 5> map_v, bsc_v, mapbp_v; 

    for (int i = 0; i<5 ; i++) {
        map_v[i] = acc_map_v[i];
        bsc_v[i] = acc_bsc_v[i];
        mapbp_v[i] = acc_mapbp_v[i];
    }

    EXPECT_EQ(ans, map_v);
    EXPECT_EQ(ans, mapbp_v);
    EXPECT_EQ(ans, bsc_v);

}


TEST(query,same_ul_mat){

    std::vector<std::string> l = {"a","b","c","d","e","f","g"};
    auto map_mat = map_prob.genAtomicRep(l);
    auto mapbp_mat = tmap_prob.genAtomicRep(l);
    auto bsc_mat = bsc_prob.genAtomicRep(l);
    
    auto map_v = map_mat.query(map_mat);
    auto bsc_v = bsc_mat.query(bsc_mat);
    auto mapbp_v = mapbp_mat.query(mapbp_mat);

    EXPECT_EQ(map_v, l);
    EXPECT_EQ(bsc_v, l);
    EXPECT_EQ(mapbp_v, l);
}

TEST(training, bundleByLabels){
    auto bsc_mat1 = bsc_prob.genUnlabeledAtomicRep(3, hd::all_true);
    auto bsc_mat2 = bsc_prob.genUnlabeledAtomicRep(3, hd::all_false);
    
    std::vector<std::string> labels = {"1","1","1","0","0","0"};

    auto bsc_mat_s = bsc_mat1.stack(bsc_mat2);

    auto AM = bsc_mat_s.bundleByLabels(labels);

    /* 
    hd::print_debug_2dbuffer(bsc_mat_s.hv_buffer);
    std::cout << std::endl;
    hd::print_debug_2dbuffer(AM.hv_buffer);
    */
    
    auto idx_ans = AM.queryIdx(bsc_mat_s);
    auto ans = AM.query(bsc_mat_s);
    auto acc_idx_ans = idx_ans.get_host_access(cl::sycl::read_only);

    std::vector idx = {0,0,0,1,1,1};

    for (int i = 0; i < idx.size(); i++)
        EXPECT_EQ( idx[i], acc_idx_ans[i]);


    EXPECT_EQ(labels,ans);
}

