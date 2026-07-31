add_executable(language language.cpp readDataset.cpp readDataset.hpp)
target_link_libraries(language PRIVATE syclhd readExeInputs ResultsHandler)
add_sycl_to_target(TARGET language SOURCES language.cpp readDataset.hpp readDataset.cpp)

add_executable(language_lse language_lse.cpp readDataset.cpp readDataset.hpp)
target_link_libraries(language_lse PRIVATE syclhd readExeInputs ResultsHandler)
add_sycl_to_target(TARGET language_lse SOURCES language_lse.cpp readDataset.hpp readDataset.cpp)