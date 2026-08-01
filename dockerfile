FROM ubuntu:24.04 AS dev

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
	build-essential \
    libboost-dev \
	cmake \
	ninja-build \
	git \
	wget \
	curl \
	python3 \
	python3-pip \
	pkg-config \
	ca-certificates \
    lsb-release \
    software-properties-common \
    gnupg

RUN wget https://apt.llvm.org/llvm.sh #Convenience script that sets up the repositories
RUN chmod +x llvm.sh
RUN ./llvm.sh 21 #Set up repositories for clang 21
RUN apt-get install -y libclang-21-dev clang-tools-21 libomp-21-dev llvm-21-dev lld-21
RUN apt-get install -y libopenblas-dev liblapack-dev liblapacke-dev 
RUN apt-get install -y doxygen
RUN rm -rf /var/lib/apt/lists/*

# Build AdaptiveCpp
WORKDIR /opt

RUN git clone --recursive https://github.com/AdaptiveCpp/AdaptiveCpp.git

WORKDIR /opt/AdaptiveCpp

RUN cmake -B build -G Ninja \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_C_COMPILER=clang-21 \
	-DCMAKE_CXX_COMPILER=clang++-21

RUN cmake --build build -j$(nproc)

RUN cmake --install build

# Project workspace
WORKDIR /workspace

CMD ["/bin/bash"]

FROM dev AS release

WORKDIR /src

COPY . .

RUN cmake -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DACPP_TARGETS="omp;cuda:sm_86" \
    -DCMAKE_INSTALL_PREFIX=/usr/local

RUN cmake --build build

RUN ctest --test-dir build

RUN cmake --install build