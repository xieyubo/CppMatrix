CppMatrix (experiment)
==

_Works for Ubuntu 24.10 (clang++-19) only._

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

## Example

### Mnist
This example is rewriting from https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/blob/master/part2_neural_network_mnist_data.ipynb.
Download the traning data from https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/refs/heads/master/mnist_dataset/mnist_train_100.csv,
and the test data from https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/refs/heads/master/mnist_dataset/mnist_test_10.csv

then run it like this:

    $ ./build/example/mnist/mnist mnist_train_100.csv mnist_test_10.csv --use-cpu

or using gpu:

    $ ./build/example/mnist/mnist mnist_train_100.csv mnist_test_10.csv --use-gpu

output:

    prediction result: 7, actual result: 7 o
    prediction result: 1, actual result: 2 x
    prediction result: 1, actual result: 1 o
    prediction result: 0, actual result: 0 o
    prediction result: 4, actual result: 4 o
    prediction result: 1, actual result: 1 o
    prediction result: 4, actual result: 4 o
    prediction result: 4, actual result: 9 x
    prediction result: 4, actual result: 5 x
    prediction result: 9, actual result: 9 o
    performance = 0.7
