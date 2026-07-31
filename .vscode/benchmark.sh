#!/bin/bash

# Get vector size and device
host=0
vector_size=1024

while getopts h:d: flag
do
    case "${flag}" in
        h) host=${OPTARG};;
        d) vector_size=${OPTARG};;
    esac
done
echo "host: $host";
echo "vector size (dimensionality): $vector_size";


# BUILD
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR && cd ..
mkdir results
cd results
rm -rf session && mkdir session
cd ../build
make examples

# RUN EXAMPLES
run() {
    ./examples/$1/$1 -vs $vector_size -host $host  >/dev/null 2>&1
}


# EMG
run emg

# MNIST
run mnist

# HDNA
# run hdna

# VoiceHD
run voicehd

# Language
run language

# Save results
cd ../results
mv session "$(date +%Y-%m-%d_%H.%M.%S)"