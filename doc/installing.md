# Installing
Instructions for getting set up with SYCL-HD.

## Dependencies
The only necessary tools to compile the SYCL-HD library are a SYCL-capable compiler + OpenBLAS (lapack.h)

Two SYCL compilers are supported:
- Intel's Data Parallel C++ 
- AdaptiveCpp

SYCL-HD predecessor DPHDC has been tested extensively on Ubuntu 20.04 LTS and Ubuntu 22.04 LTS using the Intel DPC++ compiler.

The newer version SYCL-HD was developed for the AdaptiveCpp version: `24.02.0+git.b61a1868.20240408.branch.develop` on Fedora 39, to enable the use of this compiler set `-DUSE_ACPP=ON` when running CMake, this option should be set by default.

> *To use the Intel Compiler set `-DUSE_ACPP=OFF`*

It is also necessary to have CMake version 3.13 or above. It can be easily installed on Ubunto using 

```bash
sudo apt install cmake
```


### Installing AdaptiveCpp
To install AdaptiveCpp follow the instructions on the [official repository](https://github.com/AdaptiveCpp/AdaptiveCpp).

### Installing DPC++
The Intel DPC++ compiler can be downloaded through the Intel OneAPI base toolkit (available [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)). 

__Important!__: To use this compiler set `-DUSE_ACPP=OFF` when running CMake.


## Using SYCL-HD in your project

It is recommended to use CMake in your project for maximum compatibility. Fetch this repository from your CMake configuration and make it available.

```CMake
include(FetchContent)
FetchContent_Declare(
        SYCL-HD
        GIT_REPOSITORY https://github.com/pcaires/SYCL-HD.git
)
FetchContent_MakeAvailable(SYCL-HD)

target_link_libraries(my_target PRIVATE syclhd)
```

Alternatively, if you want to link from a local version of the library, and build alongside your project, you may use something along the lines of:

```CMake
add_subdirectory(/path/to/SYCL-HD/src syclhd-src)
target_link_libraries(my_target PRIVATE syclhd)
```

Lastly, you may install the library in your machine using the install target, which will compile the library using the targets supplied to SYCL-HD.
```sh
cd build
make install
```
and then in your CMake project simply use the find_package command:
```CMake
find_package(SYCL-HD REQUIRED)
target_link_libraries(my_target PRIVATE syclhd)
```

## Docker

You may get started with using the library via docker. You may have to customize the container to enable gpu support.

```bash
sh .vscode/setup_acpp_img.sh
```
