add_executable(emg emg.cpp readData.cpp readData.hpp)
target_link_libraries(emg PRIVATE syclhd readExeInputs ResultsHandler)
add_sycl_to_target(TARGET emg SOURCES emg.cpp readData.cpp readData.hpp)

add_executable(emg_lse emg_lse.cpp readData.cpp readData.hpp)
target_link_libraries(emg_lse PRIVATE syclhd readExeInputs ResultsHandler)
add_sycl_to_target(TARGET emg_lse SOURCES emg_lse.cpp readData.cpp readData.hpp)