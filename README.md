Install:

    sudo apt install \
        libgtest-dev

Build:

    mkdir build
    cd build
    CXX=/usr/bin/clang++ cmake .. -GNinja
    ninja
