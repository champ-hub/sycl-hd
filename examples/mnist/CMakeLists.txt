include(FetchContent)
FetchContent_Declare(
    mnist_reader
        GIT_REPOSITORY https://github.com/wichtounet/mnist.git
        )
FetchContent_MakeAvailable(mnist_reader)

find_package(MNIST PATHS ${mnist_reader_SOURCE_DIR})
if(NOT MNIST_FOUND)
message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)
include_directories(${MNIST_INCLUDE_DIR})

add_executable(mnist_rbf mnist_rbf.cpp readDataset.cpp readDataset.hpp)
target_link_libraries(mnist_rbf PRIVATE syclhd readExeInputs ResultsHandler)

add_executable(mnist mnist.cpp)
target_link_libraries(mnist PRIVATE syclhd readExeInputs ResultsHandler)

add_executable(mnist_lse mnist_lse.cpp)
target_link_libraries(mnist_lse PRIVATE syclhd readExeInputs ResultsHandler)


add_sycl_to_target(TARGET mnist SOURCES mnist.cpp)
add_sycl_to_target(TARGET mnist_rbf SOURCES mnist_rbf.cpp)
add_sycl_to_target(TARGET mnist_lse SOURCES mnist_lse.cpp)