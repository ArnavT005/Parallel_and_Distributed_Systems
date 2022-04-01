#include <stdio.h>
#include <limits>
#include <algorithm>
#include "img.hpp"
#include "util.cuh"


__global__
void find(float th2, int* query_cu, int query_rows, int query_cols, float graysum_avg, int* data_cu, float* prefixsum_cu, float* result_cu) {
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
    // Point p1, p2, p3, p4;
    // p1 = rotate_point(Point{(float) col, (float) row}, Point{(float) 0, (float) 0}, angle, query_rows);
    // p2 = rotate_point(Point{(float) col, (float) row}, Point{(float) query_cols - 1, (float) 0}, angle, query_rows);
    // p3 = rotate_point(Point{(float) col, (float) row}, Point{(float) 0, (float) query_rows - 1}, angle, query_rows);
    // p4 = rotate_point(Point{(float) col, (float) row}, Point{(float) query_cols - 1, (float) query_rows - 1}, angle, query_rows);
    // float x_min, x_max, y_min, y_max;
    
    // if (angle == 0) {
    //     x_min = p1.x;
    //     x_max = p2.x;
    //     y_min = p1.y;
    //     y_max = p3.y;
    // } else if (angle == 45) {
    //     x_min = p1.x;
    //     x_max = p4.x;
    //     y_min = p2.y;
    //     y_max = p3.y;
    // } else {
    //     x_min = p3.x;
    //     x_max = p2.x;
    //     y_min = p1.y;
    //     y_max = p4.y;
    // }
    // if (!(x_min >= 0 && x_max <= gridDim.y - 1 && y_min >= 0 && y_max <= gridDim.x - 1)) {
    //     return;
    // }
    // printf("Row: %d, col: %d, rot: %d\n", row, col, rot);
    auto bb = BB{col, row, query_cols - 1, query_rows - 1};
    bb.rotate(angle);
    auto ps = get_prefix_sum(bb, gridDim.x, gridDim.y, prefixsum_cu);
    ps /= ((bb.h + 1) * (bb.w + 1));
    if( abs(ps - graysum_avg) >= th2){
        if (col == 119 && rot == 1 && row == 119 && threadIdx.x == 0) {
            printf("Gray: %f, %f\n", ps, graysum_avg);
        }
        return;
    }

    int tid = threadIdx.x;
    int chunk_sz = 1;
    if (query_rows * query_cols > blockDim.x) {
        if ((query_rows * query_cols) % blockDim.x == 0) {
            chunk_sz = (query_rows * query_cols) / blockDim.x;
        } else {
            chunk_sz = (query_rows * query_cols) / blockDim.x + 1;
        }
    }
    int start = tid * chunk_sz, end = min(start + chunk_sz, query_rows * query_cols);
    rmsd[tid] = 0.0f;
    
    for(auto query_idx = start; query_idx < end; query_idx++){
        int curr_col = query_idx % query_cols;
        int curr_row = query_idx / query_cols;
        for(auto ch=0; ch < 3; ch++){
            float data_px, query_px;
            query_px = query_cu[query_idx * 3 + ch];
            
            auto rotated_point = rotate_point(Point{(float) col, (float) row}, Point{(float) curr_col, (float) curr_row}, angle, query_rows);
            if (angle != 0) {
                data_px = bilinear_interpolate(rotated_point, ch, gridDim.x, gridDim.y, data_cu);
            } else {
                data_px = get_value(data_cu, rotated_point.y, rotated_point.x, ch, gridDim.x, gridDim.y);
            }
            rmsd[tid] += (data_px - query_px) * (data_px - query_px);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float rmsd_val = 0;
        for (int i = 0; i < blockDim.x; i ++) {
            rmsd_val += rmsd[i];
        }
        result_cu[row * gridDim.y * gridDim.z + col * gridDim.z + rot] = sqrt(rmsd_val / (query_rows * query_cols * 3));
        // if (col == 118 && rot == 1 && row == 119) {
        //     printf("118: %f\n", result_cu[row * gridDim.y * gridDim.z + col * gridDim.z + rot]);
        // }
        // if (col == 119 && rot == 1 && row == 119) {
        //     printf("119: %f\n", result_cu[row * gridDim.y * gridDim.z + col * gridDim.z + rot]);
        // }
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
    int data_rows = std::get<0>(data_sz), data_cols = std::get<1>(data_sz), data_dim = std::get<2>(data_sz);
    int query_rows = std::get<0>(query_sz), query_cols = std::get<1>(query_sz), query_dim = std::get<2>(query_sz);

    std::cout << "Data Image: " << data_rows << " " << data_cols << " " << data_dim << std::endl;
    std::cout << "Query Image: " << query_rows << " " << query_cols << " " << query_dim << std::endl;

    int data_mem = data_rows * data_cols * data_dim, query_mem = query_rows * query_cols * query_dim;

    auto gray_data_mat = rgb2gray(data_mat.get());
    auto prefixsum_mat = prefixsum(gray_data_mat.get());
    auto graysum_avg = graysum(query_mat.get());

    V<float> result_arr(data_mem, std::numeric_limits<float>::max());

    int *data_cu, *query_cu;
    float *prefixsum_cu, *result_cu;
    cudaMalloc(&data_cu, data_mem * sizeof(int)); 
    cudaMalloc(&query_cu, query_mem * sizeof(int));
    cudaMalloc(&prefixsum_cu, data_rows * data_cols * sizeof(float));
    cudaMalloc(&result_cu, data_rows * data_cols * 3 * sizeof(float));

    cudaMemcpy(data_cu, data_mat->get(), data_mem * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(query_cu, query_mat->get(), query_mem * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(prefixsum_cu, prefixsum_mat->get(), data_rows * data_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(result_cu, result_arr.data(), data_rows * data_cols * 3 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_dim = dim3(data_rows, data_cols, 3);
    dim3 block_dim = dim3(std::min(1024, query_rows * query_cols));

    // invoke kernel
    find<<<grid_dim, block_dim, block_dim.x * sizeof(float)>>>(th2, query_cu, query_rows, query_cols, graysum_avg, data_cu, prefixsum_cu, result_cu);
    cudaMemcpy(result_arr.data(), result_cu, data_mem * sizeof(float), cudaMemcpyDeviceToHost);
    
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    
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
                result_que.pop();
                result_que.push(P<int, float>(i, result_arr[i]));
            }
        }
    }
    
    V<T<int, int, int>> output(n);
    int output_sz = result_que.size(), index = output_sz - 1;
    std::ofstream fout("output.txt", std::ios::out);
    while (!result_que.empty()) {
        P<int, float> temp = result_que.top();
        result_que.pop();
        int row, col, rot;
        row = temp.first / (data_cols * data_dim);
        col = (temp.first / data_dim) % data_cols;
        rot = temp.first % data_dim;
        if (rot == 1) {
            rot = 45;
        } else if (rot == 2) {
            rot = -45;
        }
        output[index] = T<int, int, int>(data_rows - 1 - row, col, rot);
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
