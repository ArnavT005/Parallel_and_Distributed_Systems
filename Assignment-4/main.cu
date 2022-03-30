#include <stdio.h>
#include <limits>
#include <algorithm>
#include "img.hpp"

__global__
void find(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(int argc, char** argv)
{
    std::string data_img, query_img;
    double th1, th2;
    int n;
    data_img = argv[1];
    query_img = argv[2];
    th1 = std::stod(argv[3]);
    th2 = std::stod(argv[4]);
    n = std::stoi(argv[5]);
    auto data_mat = imread(data_img);
    auto query_mat = imread(query_img);
    T<int, int, int> data_sz = data_mat->shape(), query_sz = query_mat->shape();
    std::cout << "Data Image: " << std::get<0>(data_sz) << " " << std::get<1>(data_sz) << " " << std::get<2>(data_sz) << std::endl;
    std::cout << "Query Image: " << std::get<0>(query_sz) << " " << std::get<1>(query_sz) << " " << std::get<2>(query_sz) << std::endl;

    int data_mem = std::get<0>(data_sz) * std::get<1>(data_sz) * std::get<2>(data_sz);
    int query_mem = std::get<0>(query_sz) * std::get<1>(query_sz) * std::get<2>(query_sz);

    auto gray_data_mat = rbg2gray(data_mat.get());
    auto prefixsum_mat = prefixsum(gray_data_mat.get());
    auto graysum_val = graysum(query_mat.get());

    V<float> result_arr(data_mem, std::numeric_limits<float>::max());

    int *data_cu, *query_cu;
    float *prefixsum_cu, *graysum_cu, *result_cu;
    cudaMalloc(&data_cu, data_mem * sizeof(int)); 
    cudaMalloc(&query_cu, query_mem * sizeof(int)); 
    cudaMalloc(&prefixsum_cu, (data_mem / std::get<2>(data_sz)) * sizeof(float));
    cudaMalloc(&graysum_cu, sizeof(float));
    cudaMalloc(&result_cu, data_mem * sizeof(float));

    cudaMemcpy(data_cu, data_mat->get(), data_mem * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(query_cu, query_mat->get(), query_mem * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(prefixsum_cu, prefixsum_mat->get(), (data_mem / std::get<2>(data_sz)) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(graysum_cu, &graysum_val, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(result_cu, result_arr.data(), data_mem * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_dim = dim3(data_mem / std::get<2>(data_sz), 3);
    dim3 thread_dim = dim3(std::get<0>(query_sz), std::get<1>(query_sz), std::get<2>(query_sz));
    

    cudaMemcpy(result_arr.data(), result_cu, data_mem * sizeof(float), cudaMemcpyDeviceToHost);
    

    cudaFree(graysum_cu);
    cudaFree(prefixsum_cu);
    cudaFree(query_cu);
    cudaFree(data_cu);
}
