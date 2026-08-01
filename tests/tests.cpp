/**
 * @file tests.cpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief Main Tests file
 * @version 0.2
 * @date 2023-12-05
 * 
 * @copyright GPL3
 * 
 */

#include "tests.hpp"

void print_test_device() {
    auto model = hd::MAP(1);
    
    std::cout << "Using device "
              << model.getDeviceName() 
              << std::endl;

    std::cout << "CPU device: " << cl::sycl::queue{cl::sycl::cpu_selector_v}.get_device().get_info<cl::sycl::info::device::name>() << std::endl;
    //std::cout << "GPU device: " << cl::sycl::queue{cl::sycl::gpu_selector_v}.get_device().get_info<cl::sycl::info::device::name>() << std::endl;
}

int main(int argc, char **argv) {
    print_test_device();
    testing::InitGoogleTest(&argc, argv);
    auto r = RUN_ALL_TESTS();
    print_test_device();
    return r;
}