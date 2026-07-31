add_executable(hdna_ hdnaBats.cpp readDataset.cpp readDataset.hpp)
target_link_libraries(hdna_ PRIVATE syclhd readExeInputs ResultsHandler)
add_sycl_to_target(TARGET hdna_ SOURCES hdnaBats.cpp)

add_executable(hdna hdnaSplice.cpp readDataset.cpp readDataset.hpp)
target_link_libraries(hdna PRIVATE syclhd readExeInputs ResultsHandler)
add_sycl_to_target(TARGET hdna SOURCES hdnaSplice.cpp)

add_executable(hdna_lse hdna_lse.cpp readDataset.cpp readDataset.hpp)
target_link_libraries(hdna_lse PRIVATE syclhd readExeInputs ResultsHandler)
add_sycl_to_target(TARGET hdna_lse SOURCES hdna_lse.cpp)