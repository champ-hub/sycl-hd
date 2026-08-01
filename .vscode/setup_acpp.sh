set -e
rm -rf build
mkdir build && cd build
cmake .. -DACPP_TARGETS="generic" -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cd tests && make tests && ctest