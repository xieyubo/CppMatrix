C++ Neural Network
==

## Install Dependencies:

    sudo apt install \
        clang \
        cmake \
        libgtest-dev \
        ninja-build

## Build:

    mkdir build
    cd build
    CXX=clang++ cmake .. -GNinja
    ninja
