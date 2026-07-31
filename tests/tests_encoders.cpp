/**
 * @file tests_encoders.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-13
 * 
 * @copyright GPL3
 * 
 */

#include "tests.hpp"
#ifndef DPHDC_TESTS_HPP
#error Tests not defined!
#endif

TEST(tests_encoders, n_gram_map) {

    const int hd_vector_size = 3;

    hd::model model = hd::MAP(hd_vector_size);

    auto rep = model.genUnlabeledAtomicRep(3,hd::all_zero);

    {
        auto acc = rep.getBufferAcessor();
        acc[0][0] = 1 , acc[1][1] = 1 , acc[2][2] = 1;
        /********
        * 1 0 0 *
        * 0 1 0 * 
        * 0 0 1 *
        *********/
    }

    std::vector<std::vector<size_t>> enc1 = {
        {1,0,2} 
    };
    std::vector<std::vector<size_t>> enc2 = {
        {1,2,0} 
    };

    auto m1 = rep.ngram(enc1, 3, hd::shift_right);
    auto m2 = rep.ngram(enc2, 3, hd::shift_left);

    std::vector<std::vector<float>> ans = {{0, 3, 0}};

    auto shifted_right = m1.getVectors();
    auto shifted_left = m2.getVectors();

    EXPECT_EQ(shifted_right, ans);
    EXPECT_EQ(shifted_left, ans);

}

TEST(tests_encoders, n_gram_mapbp) {

    const int hd_vector_size = 4*3;

    hd::model model = hd::TMAP(hd_vector_size);

    auto rep1 = model.genUnlabeledAtomicRep(1,hd::all_zero);
    auto rep2 = model.genUnlabeledAtomicRep(1,hd::all_true);
    auto rep3 = model.genUnlabeledAtomicRep(1,hd::all_false);
    auto rep4 = model.genUnlabeledAtomicRep(3);

    hd::matrix<hd::ternary_hv_buffer> rep = rep1.stack(rep2).stack(rep3).stack(rep4);

    /* 
    std::cout << "Zero:  ";
    hd::print_debug_2dbuffer(rep1.hv_buffer);
    std::cout << "True:  ";
    hd::print_debug_2dbuffer(rep2.hv_buffer);
    std::cout << "False: ";
    hd::print_debug_2dbuffer(rep3.hv_buffer);
    */

    std::vector<std::vector<size_t>> enc1 = {
        {1,0,2} 
    };
    std::vector<std::vector<size_t>> enc2 = {
        {1,2,0} 
    };

    auto m1 = rep.ngram(enc1, 3, hd::shift_right);
    auto m2 = rep.ngram(enc2, 3, hd::shift_left);
    
    /* 
    hd::print_debug_2dbuffer(rep.hv_buffer);
    hd::print_debug_2dbuffer(m1.hv_buffer);
    hd::print_debug_2dbuffer(m2.hv_buffer);
    */

    auto m1v = m1.getVectors();
    auto m2v = m2.getVectors();

    EXPECT_EQ(m1v,m2v);

}

TEST(tests_encoders, n_gram_bsc_left_right) {

    const int hd_vector_size = 8*3;

    hd::model model = hd::BSC(hd_vector_size);

    auto rep = model.genUnlabeledAtomicRep(3,hd::random);

    std::vector<std::vector<size_t>> enc1 = {
        {1,0,2} 
    };
    std::vector<std::vector<size_t>> enc2 = {
        {1,2,0} 
    };

    auto m1 = rep.ngram(enc1, 3, hd::shift_right);
    auto m2 = rep.ngram(enc2, 3, hd::shift_left);

    auto m1v = m1.getVectors();
    auto m2v = m2.getVectors();

    EXPECT_EQ(m1v,m2v);

}


TEST(tests_encoders, n_gram_barelshift) {

    std::vector<std::vector<size_t>> enc1 = {
        {0,1,2,3,4,5,6,7,8} 
    };

    std::vector<std::vector<size_t>> res1_ans_true = {
        {
            0,1,2,
            1,2,3,
            2,3,4,
            3,4,5,
            4,5,6,
            5,6,7,
            6,7,8
        } 
    };

    auto res1 = hd::barrelshift(enc1, 3,9);
    auto acc_res1 = res1.get_host_access(cl::sycl::read_only);

    std::vector<std::vector<size_t>> res1_ans = {std::vector<size_t>(acc_res1.begin(),acc_res1.end())};
    

    EXPECT_EQ(res1_ans_true,res1_ans);

}

TEST(tests_encoders, baselevel_ngram_map){

    auto model = hd::MAP(4);
    std::vector<std::vector<float>> b = {
        {1,1,1,1},
        {0,1,2,3},
        {-1,1,-1,1}
    };

    std::vector<std::vector<float>> l = {
        {1,1,1,1},
        {-1,1,-1,1},
        {0,0,0,0},
        {0,1,2,3}
    };

    auto bm = model.genUnlabeledAtomicRep(3,hd::none);
    auto lm = model.genUnlabeledAtomicRep(4,hd::none);
    bm.setVectors(b);
    lm.setVectors(l);

    std::vector<std::vector<uint>> d = {
        {2,2,2},
        {1,2,2},
        {2,3,2}
    };

    std::vector<std::vector<float>> ans = {
        l[2],
        l[1],
        {9,0,1,4}
    };


    auto enc = bm.ngramBaseLevel(lm,d,2,hd::shift_right);
    auto res = enc.getVectors();

    EXPECT_EQ(res, ans);
}

TEST(RBF, methods) {
    const int hd_vector_size = 10;

    hd::model model = hd::MAP(hd_vector_size);

    std::vector<std::vector<float>> feature_vecs = {
        {1,2,3,4},
        {1,2,3,4},
        {-1,-2,-3,-4}
    };

    hd::matrix<hd::float_2d_buffer> rep1 = model.RBFKernelTrick(feature_vecs,0,1,hd::cos_dist);
    hd::matrix<hd::float_2d_buffer> rep2 = model.RBFKernelTrick(feature_vecs,0,1,hd::cos_dot);
    hd::matrix<hd::float_2d_buffer> rep3 = model.RBFKernelTrick(feature_vecs);

    //hd::print_debug_2dbuffer(rep1.hv_buffer);
    //std::cout << std::endl;
    //hd::print_debug_2dbuffer(rep2.hv_buffer);
    //std::cout << std::endl;
    //hd::print_debug_2dbuffer(rep3.hv_buffer);

    EXPECT_EQ(cl::sycl::range<2>(3,hd_vector_size), rep1.hv_buffer.get_range());
    EXPECT_EQ(rep1.hv_buffer.get_range(), rep2.hv_buffer.get_range());
    EXPECT_EQ(rep2.hv_buffer.get_range(), rep3.hv_buffer.get_range());
}

TEST(RBF, cos_dist_map) {

    const int hd_vector_size = 10;

    hd::model model = hd::MAP(hd_vector_size);

    std::vector<std::vector<float>> feature_vecs = {
        {1,2,3,4},
        {1,2,3,4},
        {-1,-2,-3,-4}
    };

    hd::matrix<hd::float_2d_buffer> rep1 = model.RBFKernelTrick(feature_vecs,0,1,hd::cos_dist);
    hd::matrix<hd::float_2d_buffer> rep2 = model.RBFKernelTrick(feature_vecs,0,1,hd::cos_dist);

    hd::float_2d_buffer sim_mat_buff = rep1.distanceMatrix(rep2);

    //hd::print_debug_2dbuffer(sim_mat_buff);

    auto acc = sim_mat_buff.get_host_access(cl::sycl::read_only);

    const std::vector<std::vector<float>> ans = {
        {1,1,-1},
        {1,1,-1},
        {-1,-1,1},
    };

    for(auto i = 0; i < ans.size(); i++)
        for(auto j = 0; j < ans[0].size(); j++)
            EXPECT_FLOAT_EQ(acc[i][j],ans[i][j]);
}

TEST(RBF, maniHD){
    std::vector<std::vector<float>> B = {
        {-2,3,2,-3},
        {0,-1,2,3},
    };
    std::vector<std::vector<float>> FV ={
        {1,2,3,4},
        {1,2,3,4},
        {0.2,0.6,1,90}
    };

    auto B_buff = hd::mat2RectBuff(B);
    auto FV_buff = hd::mat2RectBuff(FV);

    auto q = hd::MAP(1).getQueue();

    auto res_m = hd::matrix<hd::float_2d_buffer>(
        hd::maniHDRBF(FV_buff,B_buff,q),
        q
    );

    hd::print_debug_2dbuffer(
        res_m.hv_buffer
    );
    
    auto res = res_m.getVectors();

    std::vector<std::vector<float>> ans(FV.size(),std::vector<float>(B.size(),0));
    for(auto i = 0; i < FV.size(); i++)
        for(auto j = 0; j < B.size(); j++){
            for(auto k = 0; k < FV[i].size(); k++)
                ans[i][j] += FV[i][k] * B[j][k];
            ans[i][j] = cos(ans[i][j]); 
            EXPECT_FLOAT_EQ(res[i][j],ans[i][j]);
        }
    //EXPECT_FLOAT_EQ(0,-1);
}

TEST(RBF, modneuralHD){
    std::vector<std::vector<float>> B = {
        {-2,3,2,-3},
        {0,-1,2,3},
    };
    std::vector<std::vector<float>> FV ={
        {1,2,3,4},
        {1,2,3,4},
        {0.8,2,-2,-1}
    };

    auto B_buff = hd::mat2RectBuff(B);
    auto FV_buff = hd::mat2RectBuff(FV);

    auto q = hd::MAP(1).getQueue();

    auto res_m = hd::matrix<hd::float_2d_buffer>(
        hd::modneuralHDRBF(FV_buff,B_buff,q),
        q
    );
    auto res = res_m.getVectors();

    std::vector<std::vector<float>> ans(FV.size(),std::vector<float>(B.size(),0));
    for(auto i = 0; i < FV.size(); i++)
        for(auto j = 0; j < B.size(); j++){
            for(auto k = 0; k < FV[i].size(); k++)
                ans[i][j] += FV[i][k] * B[j][k];
            ans[i][j] = cos(ans[i][j]) * sin(ans[i][j]); 
            EXPECT_FLOAT_EQ(res[i][j],ans[i][j]);
        }
    
    //hd::print_debug_2dbuffer(res_m.hv_buffer);
}
/* 
TEST(tests_encoders, bundle){

    auto m = hd::BSC(8);

    std::vector<std::vector<uint>> a = {{0,1},{2,3}};
    auto a_i = hd::imat2op(a);

    hd::print_debug_1dbuffer(a_i.indexes);
    hd::print_debug_1dbuffer(a_i.new_entry); 

    auto m1 = m.genUnlabeledAtomicRep(3,hd::all_false);
    auto m2 = m.genUnlabeledAtomicRep(3,hd::all_true);

    auto m3 = m1.stack(m2);

    auto res = m3.bundle(a_i).getVectors();

    for (auto i = 0; i < res.size(); i++){
        for (auto j = 0; j < res[i].size(); j++){
            EXPECT_EQ(res[i][j],hd::b8_char{i});
        }
    }

} */