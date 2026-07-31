add_executable(cifar10 cifar10.cpp cifar10_reader.hpp)
target_link_libraries(cifar10 PRIVATE syclhd ResultsHandler)
add_sycl_to_target(TARGET cifar10)


add_executable(cifar10_lse cifar10_lse.cpp cifar10_reader.hpp)
target_link_libraries(cifar10_lse PRIVATE syclhd ResultsHandler)
add_sycl_to_target(TARGET cifar10_lse)
