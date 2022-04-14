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
typedef long long ll;
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
template class matrix<ll>;

matrix<int>* imread(string);
float graysum(matrix<int>*);
matrix<ll>* prefixsum(matrix<int>*);
