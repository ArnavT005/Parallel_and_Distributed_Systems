#include "img.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::string data_img, query_img;
    double th1, th2;
    int n;
    data_img = argv[1];
    query_img = argv[2];
    th1 = std::stod(argv[3]);
    n = std::stoi(argv[4]);
    matrix<0, 0, 0> data_mat, query_mat;
    read_img(data_img, data_mat);
    read_img(query_img, query_mat);
    T<int, int, int> data_sz = data_mat.shape(), query_sz = query_mat.shape();
    std::cout << "Data Image: " << std::get<0>(data_sz) << " " << std::get<1>(data_sz) << " " << std::get<2>(data_sz) << std::endl;
    std::cout << "Query Image: " << std::get<0>(query_sz) << " " << std::get<1>(query_sz) << " " << std::get<2>(query_sz) << std::endl;
}