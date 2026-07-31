/**
 * @file tests_lse.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-10-15
 * 
 * @copyright GPL3
 * 
 */

#include "tests.hpp"
#include <syclhd.hpp>
#ifndef DPHDC_TESTS_HPP
#error Tests not defined!
#endif

static std::vector<std::vector<float>> mat = {
    {1,0.1},
    {-0.1,-1},
    {-0.5,-1}
};

static std::vector<std::vector<float>> mat1 = {
    {3,0.1,0.1},
    {-0.1,-6, 0.1},
    {-0.5,-2, -0.3},
};

static cl::sycl::queue q{};

TEST(tests_lse, corr_mat) {
    auto H = hd::mat2RectBuff(mat);
    auto corr_mat = hd::lse::corr_matrix(H, q);
    auto acc = corr_mat.get_host_access(cl::sycl::read_only);

    std::vector<std::vector<float>> result = {
        {1.26, 0.7},
        {0.7, 2.01}
    };

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            ASSERT_FLOAT_EQ(result[i][j], acc[i][j]);
        }
    }
}

TEST(tests_lse, sum_i) {
    auto H = hd::mat2RectBuff(mat);
    auto corr_mat = hd::lse::corr_matrix(H, q);
    auto corr_mat1 = hd::lse::corr_matrix(H, q);
    
    hd::lse::plus_i(corr_mat, q);

    auto acc = corr_mat.get_host_access(cl::sycl::read_only);
    auto res = corr_mat1.get_host_access(cl::sycl::read_only);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            ASSERT_FLOAT_EQ(res[i][j] + (float) (i==j), acc[i][j]);
        }
    }
}



TEST(tests_lse, inv) {

    std::vector<std::vector<float>> result = {
        {10050/10213.0, -500/1459.0},
        {-500/1459.0, 900/1459.0}
    };

    auto H = hd::mat2RectBuff(mat);
    auto corr_mat = hd::lse::corr_matrix(H, q);
    hd::lse::inv(corr_mat);

    auto acc = corr_mat.get_host_access(cl::sycl::read_only);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            ASSERT_FLOAT_EQ(result[i][j], acc[i][j]);
        }
    }


}

TEST(tests_lse, inv_sym) {


    auto H = hd::mat2RectBuff(mat);
    auto corr_mat = hd::lse::corr_matrix(H, q);
    auto corr_mat1 = hd::lse::corr_matrix(H, q);


    hd::lse::inv(corr_mat);
    hd::lse::inv_sym(corr_mat);

    auto acc = corr_mat.get_host_access(cl::sycl::read_only);
    auto res = corr_mat1.get_host_access(cl::sycl::read_only);

    for (int i = 0; i < 2; i++) {
        for (int j = i; j < 2; j++) {
            ASSERT_FLOAT_EQ(acc[i][j], res[i][j]);
        }
    }


}

TEST(tests_lse, proj) {

    auto H = hd::mat2RectBuff(mat);
    auto Y = hd::mat2RectBuff(mat1);
    auto proj_mat = hd::lse::projection(H, Y, q);
    auto acc = proj_mat.get_host_access(cl::sycl::read_only);
    std::vector<std::vector<float>> result = {
        {163/50.0, 17/10.0, 6/25.0},
        {9/10.0, 801/100.0, 21/100.0}
    };

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            ASSERT_FLOAT_EQ(result[i][j], acc[i][j]);
        }
    }
}

TEST(tests_lse, mult) {
    auto H = hd::mat2RectBuff(mat);
    auto corr_mat = hd::lse::corr_matrix(H, q);
    auto Y = hd::mat2RectBuff(mat1);
    auto proj_mat = hd::lse::projection(H, Y, q);
    hd::lse::inv(corr_mat);
    auto res = hd::lse::mult(corr_mat, proj_mat, q);

    auto acc = res.get_host_access(cl::sycl::read_only);

    std::vector<std::vector<float>> result = {
        {29613/10213.0,-10950/10213.0, 1677/10213.0},
        {-820/1459.0, 6359/1459.0, 69/1459.0}
    };

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            ASSERT_FLOAT_EQ(result[i][j], acc[i][j]);
        }
    }
}