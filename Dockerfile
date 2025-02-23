ARG UBUNTU_VERSION
FROM ubuntu:${UBUNTU_VERSION}

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
        clang \
        clang-tools \
        cmake \
        libgtest-dev \
        ninja-build \
        nlohmann-json3-dev\

WORKDIR /src

COPY . .

RUN mkdir build && \
    cd build && \
    CXX=clang++ cmake .. -GNinja

RUN cd build && \
    ninja