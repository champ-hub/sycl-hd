include(CTest)

### Downloading Google Test and making it available
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

set(tests_cpp tests.cpp
              tests_general.cpp
              tests_encoders.cpp
              tests_regen.cpp
              tests_operators.cpp
              tests_lse.cpp
              tests.hpp)

add_executable(tests ${tests_cpp})
add_dependencies(tests syclhd gtest)
target_link_libraries(tests PRIVATE syclhd gtest)

add_sycl_to_target(TARGET tests SOURCES ${tests_cpp})

add_test(NAME tests COMMAND tests)