### Basic Project info ###
cmake_minimum_required(VERSION 3.13.0)
set(CMAKE_CXX_STANDARD 17)
project(SYCL-HD VERSION 0.3 DESCRIPTION "A SYCL based framework for Hyperdimensional Computing")

### Find SYCL Compiler ###
OPTION(USE_ACPP "Use AdaptiveCpp Compiler" ON)
IF(USE_ACPP)
find_package(AdaptiveCpp REQUIRED)
set(CMAKE_CXX_FLAGS "-Wno-unknown-cuda-version") # -pg") #Profile flag and ignore unknown cuda version warningXXx
ELSE()
find_package(IntelSYCL CONFIG REQUIRED)
add_link_options(-lOpenCL -lpthread -lm -ldl)
ENDIF()

### Find OpenBLAS ###
set(BLA_VENDER OpenBLAS)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)


### Making project variables available in code ###
add_definitions(-DPROJECT_PATH_CMAKE=\"${PROJECT_SOURCE_DIR}\")
add_definitions(-DPROJECT_VERSION_CMAKE=\"${CMAKE_PROJECT_VERSION}\")
add_definitions(-DPROJECT_NAME_CMAKE=\"${CMAKE_PROJECT_NAME}\")


### Adding directories ###
add_subdirectory(src/)

OPTION(BUILD_WITH_TESTS "Include tests when building" ON)
IF(BUILD_WITH_TESTS)
	add_subdirectory(tests/)
ENDIF()

OPTION(BUILD_WITH_EXAMPLES "Include examples when building" ON)
IF(BUILD_WITH_EXAMPLES)
	add_subdirectory(examples/)
ENDIF()

OPTION(BUILD_WITH_DOCS "Include option to make docs" ON)
IF(BUILD_WITH_DOCS)
	add_subdirectory(doc/)
ENDIF()