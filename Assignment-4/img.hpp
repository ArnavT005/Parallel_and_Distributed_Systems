#pragma once
#include <tuple>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

typedef unsigned int uint;

template <typename T>
using V = std::vector<T>;

template <typename I, typename J, typename K>
using T = std::tuple<I, J, K>;

class matrix {
    private:
        V<V<V<int>>> mat;
        int row, col, dim;
    public:
        matrix();
        matrix(int, int, int);
        V<V<int>> operator[](int);
        void resize(int, int, int, bool init = false, int val = 0);
        T<int, int, int> shape();
};

void read_img(std::string, matrix&);