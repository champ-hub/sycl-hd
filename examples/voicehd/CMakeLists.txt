add_executable(voicehd voicehd.cpp readDataset.cpp readDataset.hpp)
target_link_libraries(voicehd PRIVATE syclhd readExeInputs ResultsHandler)
add_sycl_to_target(TARGET voicehd SOURCES voicehd.cpp readDataset.cpp readDataset.hpp)


add_executable(voicehd_lse voicehd_lse.cpp readDataset.cpp readDataset.hpp)
target_link_libraries(voicehd_lse PRIVATE syclhd readExeInputs ResultsHandler)
add_sycl_to_target(TARGET voicehd_lse SOURCES voicehd_lse.cpp readDataset.cpp readDataset.hpp)