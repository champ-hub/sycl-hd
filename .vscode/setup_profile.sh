SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

if [0 = 1]; then

cd ..
rm -rf build-prof
mkdir build-prof && cd build-prof
cmake .. -DACPP_TARGETS="omp;cuda:sm_86" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg
fi

# Profile EMG for example
cd $SCRIPT_DIR && cd ../build-prof

PROFILE_CMD() {
    gp-collect-app -O ../results/test.er -p on -S on $@
}

make mnist
PROFILE_CMD ./examples/mnist/mnist -vs 4000 -host 1
gp-display-gui ../results/test.er
