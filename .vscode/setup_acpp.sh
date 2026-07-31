SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

cd ..
rm -rf build
mkdir build && cd build
cmake .. -DACPP_TARGETS="omp;cuda:sm_86" -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cd tests && make tests && ctest