SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ../$SCRIPT_DIR

rm -rf build-intel
mkdir build-intel && cd build-intel

#source /opt/intel/oneapi/setvars.sh --include-intel-llvm

cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_ACPP=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DBUILD_WITH_DOCS=OFF -DCMAKE_CXX_COMPILER=icpx
