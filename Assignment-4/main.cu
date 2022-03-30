#include <stdio.h>
#include <limits>
#include <algorithm>
#include "img.hpp"
#include "util.cuh"


__global__
void find(float th1_cu, float th2_cu, int threadDimX, int threadDimY, int threadDimZ, int* data_cu, int* query_cu, float* prefixsum_cu, float graysum_cu, float* result_cu) {
    extern __shared__ float rmsd[];
    int row, col, rot, angle;
    row = blockIdx.x;
    col = blockIdx.y; 
    rot = blockIdx.z;
    if (rot == 0) {
        angle = 0;
    } else if (rot == 1) {
        angle = 45;
    } else {
        angle = -45;
    }
    printf("Row: %d, col: %d, rot: %d\n", row, col, rot);
    // filter
    
    //calculate rmsd
    int dim = threadIdx.z;
    Point rotated_point;
    float data_px, query_px;
    query_px = query_cu[threadIdx.x * threadDimY * threadDimZ + threadIdx.y * threadDimZ + threadIdx.z];
    rotated_point = rotate_point(Point{(float) col, (float) row}, Point{(float) threadIdx.y, (float) threadIdx.x}, angle);
    if (angle != 0) {
        data_px = bilinear_interpolate(rotated_point, blockDim.x, blockDim.y, data_cu);
    } else {
        data_px = get_value(data_cu, rotated_point.y, rotated_point.x, blockDim.x, blockDim.y);
    }
    rmsd[threadIdx.x * threadDimY * threadDimZ + threadIdx.y * threadDimZ + threadIdx.z] = (data_px - query_px) * (data_px - query_px);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        float rmsd_val = 0;
        for (int i = 0; i < threadDimX * threadDimY * threadDimZ; i ++) {
            rmsd_val += rmsd[i];
        }
        result_cu[row * blockDim.y * blockDim.z + col * blockDim.z + rot] = sqrt(rmsd_val / (threadDimX * threadDimY * threadDimZ));
    }
}

struct comparePairMax {
    bool operator() (const P<int, float>& a, const P<int, float>& b) {
        if (a.second < b.second) {
            return true;
        } else if (a.second == b.second) {
            return a.first < b.first;
        } else {
            return false;
        }
    }
};

int main(int argc, char** argv) {
    std::string data_img, query_img;
    float th1, th2;
    int n;
    data_img = argv[1];
    query_img = argv[2];
    th1 = std::stof(argv[3]);
    th2 = std::stof(argv[4]);
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
    float *prefixsum_cu, *result_cu;
    cudaMalloc(&data_cu, data_mem * sizeof(int)); 
    cudaMalloc(&query_cu, query_mem * sizeof(int));
    cudaMalloc(&prefixsum_cu, (data_mem / std::get<2>(data_sz)) * sizeof(float));
    cudaMalloc(&result_cu, data_mem * sizeof(float));

    cudaMemcpy(data_cu, data_mat->get(), data_mem * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(query_cu, query_mat->get(), query_mem * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(prefixsum_cu, prefixsum_mat->get(), (data_mem / std::get<2>(data_sz)) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(result_cu, result_arr.data(), data_mem * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_dim = dim3(std::get<0>(data_sz), std::get<1>(data_sz), 3);
    dim3 thread_dim = dim3(std::get<0>(query_sz), std::get<1>(query_sz), std::get<2>(query_sz));
    
    //Invoke Kernel
    saxpy<<<block_dim, dim3(10,10,10)>>>();
    auto err = cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Device Synchronize: " << cudaGetErrorString(err) << std::endl;
    }
    // find<<<block_dim, thread_dim, query_mem * sizeof(float)>>>(th1, th2, std::get<0>(query_sz), std::get<1>(query_sz), std::get<2>(query_sz), data_cu, query_cu, prefixsum_cu, graysum_val, result_cu);
    cudaMemcpy(result_arr.data(), result_cu, data_mem * sizeof(float), cudaMemcpyDeviceToHost);
    
    
    std::priority_queue<P<int, float>, V<P<int, float>>, comparePairMax> result_que;
    for (int i = 0; i < result_arr.size(); i ++) {
        if (result_arr[i] >= th1) {
            continue;
        }
        if (result_que.size() < n) {
            result_que.push(P<int, float>(i, result_arr[i]));
        } else {
            if (result_arr[i] >= result_que.top().second) {
                continue;
            } else {
                result_que.push(P<int, float>(i, result_arr[i]));
                result_que.pop();
            }
        }
    }
    
    V<T<int, int, int>> output(n);
    int index = result_que.size() - 1, output_sz = index + 1;
    std::ofstream fout("output.txt", std::ios::out);
    while (!result_que.empty()) {
        P<int, float> temp = result_que.top();
        result_que.pop();
        int row, col, rot;
        row = temp.first / (std::get<1>(data_sz) * std::get<2>(data_sz));
        col = (temp.first / (std::get<2>(data_sz))) % std::get<1>(data_sz);
        rot = temp.first % std::get<2>(data_sz);
        if (rot == 1) {
            rot = 45;
        } else if (rot == 2) {
            rot = -45;
        }
        output[index] = T<int, int, int>(row, col, rot);
        index --;
    }
    for (int i = 0; i < output_sz; i ++) {
        fout << std::get<0>(output[i]) << " " << std::get<1>(output[i]) << " " << std::get<2>(output[i]) << std::endl;
    }
    fout.close();
    cudaFree(result_cu);
    cudaFree(prefixsum_cu);
    cudaFree(query_cu);
    cudaFree(data_cu);
}
