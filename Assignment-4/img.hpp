#pragma once
#include <tuple>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include <assert.h>

typedef unsigned int uint;
using namespace std;

template <typename T, typename K>
using P = std::pair<T, K>;

template <typename I, typename J, typename K>
using T = std::tuple<I, J, K>;

template <typename T>
using V = std::vector<T>;

template <typename K>
class matrix {
    private:
        K* mat;
        int row, col, dim;
    public:
        matrix();
        matrix(int, int, int);
        void set(int, int, int, K&);
        K* get();
        K get(int, int, int);
        void resize(int, int, int, bool init = false, K val = K());
        T<int, int, int> shape();
};

template class matrix<int>;
template class matrix<float>;

// unique_ptr<matrix<int>> imread(string);
// unique_ptr<matrix<float>> rgb2gray(matrix<int>*);
// float graysum(matrix<int>*);

// // Prefix sum for 2D matrices
// unique_ptr<matrix<float>> prefixsum(matrix<float>*);
matrix<int>* imread(string);
matrix<float>* rgb2gray(matrix<int>*);
float graysum(matrix<int>*);

// Prefix sum for 2D matrices
matrix<float>* prefixsum(matrix<float>*);
