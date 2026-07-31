/**
 * @file tests_operators.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief Test Operators
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

TEST(operator, mapbp_index){
    auto model = hd::TMAP(4);
    auto mat_t = model.genUnlabeledAtomicRep(1,hd::all_true).getVectors();
    auto mat_f = model.genUnlabeledAtomicRep(1,hd::all_false).getVectors();
    auto mat_z = model.genUnlabeledAtomicRep(1,hd::all_zero).getVectors();
    auto mat_r = model.genUnlabeledAtomicRep(1,hd::random).getVectors();

    for (int i = 0; i < 4; i++){
        EXPECT_EQ(mat_t[0][0][i],-1);
        EXPECT_EQ(mat_f[0][0][i],1);
        EXPECT_EQ(mat_z[0][0][i],0);
        EXPECT_NE(mat_r[0][0][i],0);
    }
}

TEST(operator, bsc_index){
    auto model = hd::BSC(8);
    auto mat_t = model.genUnlabeledAtomicRep(1,hd::all_true).getVectors();
    auto mat_f = model.genUnlabeledAtomicRep(1,hd::all_false).getVectors();

    for (int i = 0; i < 8; i++){
        EXPECT_EQ(mat_t[0][0][i],1);
        EXPECT_EQ(mat_f[0][0][i],0);
    }
}