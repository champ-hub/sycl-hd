add_subdirectory(hdna/)
add_subdirectory(mnist/)
add_subdirectory(language/)
add_subdirectory(voicehd/)
add_subdirectory(emg/)
add_subdirectory(dofmex/)
add_subdirectory(cifar10/)

add_library(readExeInputs STATIC
        readExeInputs.cpp
        readExeInputs.hpp)
target_include_directories(readExeInputs BEFORE INTERFACE .)

add_library(ResultsHandler STATIC
        ResultsHandler.cpp
        ResultsHandler.hpp)
target_link_libraries(ResultsHandler PRIVATE syclhd readExeInputs)
add_sycl_to_target(TARGET ResultsHandler SOURCES ResultsHandler.cpp ResultsHandler.hpp)
target_include_directories(ResultsHandler BEFORE INTERFACE .)

add_custom_target(examples DEPENDS hdna hdna_ mnist language voicehd emg dofmex cifar10)
add_custom_target(examples_lse DEPENDS hdna_lse mnist_lse language_lse voicehd_lse emg_lse cifar10_lse)