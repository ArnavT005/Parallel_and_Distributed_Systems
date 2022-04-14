#include <stdio.h>
#include <limits>
#include <algorithm>
#include <vector>

#include <tuple>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include <assert.h>
#include "util.cuh"


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


template <typename K>
matrix<K>::matrix() {
    mat = nullptr;
    row = col = dim = 0;
}

template <typename K>
matrix<K>::matrix(int R, int C, int k) {
    mat = new K[R * C * k];
    row = R;
    col = C;
    dim = k;
}

template <typename K>
void matrix<K>::set(int i, int j, int k, K& val) {
    mat[i * col * dim + j * dim + k] = val;
}

template <typename K>
K* matrix<K>::get() {
    return mat;
}

template <typename K>
K matrix<K>::get(int i, int j, int k) {
    return mat[i * col * dim + j * dim + k];
}


template <typename K>
void matrix<K>::resize(int row, int col, int dim, bool init, K val) {
    if (mat != nullptr) {
        delete mat;
    }
    mat = new K[row * col * dim];
    if (init) {
        for (int i = 0; i < row; i ++) {
            for (int j = 0; j < col; j ++) {
                for (int k = 0; k < dim; k ++) {
                    if (init) {
                        set(i, j, k, val);
                    }
                    
                }
            }
        }
    }
    this->row = row;
    this->col = col;
    this->dim = dim;
}

template <typename K>
T<int, int, int> matrix<K>::shape() {
    return std::make_tuple(row, col, dim);
}

matrix<int>* imread(string file_img) {
    std::string line;
    std::ifstream fin(file_img, std::ios::in);
    std::getline(fin, line);
    std::stringstream stream(line);
    int row, col;
    stream >> row >> col;
    auto img_mat = new matrix<int>(row, col, 3);
    std::getline(fin, line);
    stream.str("");
    stream.clear();
    stream << line;
    for (int i = 0; i < row; i ++) {
        for (int j = 0; j < col; j ++) {
            int R, G, B;
            stream >> R >> G >> B;
            img_mat->set(i, j, 0, R);
            img_mat->set(i, j, 1, G);
            img_mat->set(i, j, 2, B);
        }
    }
    return img_mat;
}

float graysum(matrix<int>* img){
    auto shape = img->shape();
    auto row = std::get<0>(shape);
    auto col = std::get<1>(shape);
    auto dim = std::get<2>(shape);
    ll temp = 0;
    for (int i = 0; i < row; i ++) {
        for (int j = 0; j < col; j ++) {
            ll dim_sum = 0;
            for(int k = 0; k < dim; k++){
                dim_sum += (ll) img->get(i,j,k);
            }
            temp += dim_sum;
        }
    }
    float sum = temp / ((float) dim * row * col);
    return sum;
}


matrix<ll>* prefixsum(matrix<int>* img){
    auto shape = img->shape();
    auto row = std::get<0>(shape);
    auto col = std::get<1>(shape);
    auto dim = std::get<2>(shape);

    assert(dim == 3);

    auto ps = new matrix<ll>();
    ps->resize(row, col, 1, true, 0);
    for (int i = 0; i < row; i ++) {
        for (int j = 0; j < col; j ++) {
            ll val = (ll) img->get(i, j, 0) + img->get(i, j, 1) + img->get(i, j, 2);
            ll sum = 0;
            if (i != 0 && j != 0){
                sum = val + ps->get(i-1, j, 0) + ps->get(i, j-1, 0) - ps->get(i-1,j-1,0);
            } else {
                if (i != 0) {
                    sum = val + ps->get(i-1, j, 0);
                } else {
                    if (j != 0) {
                        sum = val + ps->get(i, j-1, 0);
                    } else {
                        sum = val;
                    }
                }
            }
            ps->set(i, j, 0, sum);
        }
    }
    return ps;
}

__global__
void find(float th2, int* query_cu, int query_rows, int query_cols, float graysum_avg, int* data_cu, int data_rows, int data_cols, ll* prefixsum_cu, float* result_cu, float* graydiff_cu, float* query_rotated_1_cu, float* query_rotated_2_cu) {
    int bid = blockIdx.x;
    int rot = blockIdx.y;
    int block_sz = 1, angle, row, col;
    if (data_rows * data_cols > gridDim.x) {
        if ((data_rows * data_cols) % gridDim.x == 0) {
            block_sz = (data_rows * data_cols) / gridDim.x;
        } else {
            block_sz = (data_rows * data_cols) / gridDim.x + 1;
        }
    }
    auto start_block = bid * block_sz, end_block = min(start_block + block_sz, data_rows * data_cols);
    float data_px[3]{0, 0, 0}, query_px[3]{0, 0, 0};
    for (int block_idx = start_block + threadIdx.x; block_idx < end_block; block_idx += blockDim.x) {
        row = block_idx / data_cols;
        col = block_idx % data_cols;
        if (rot == 0) {
            angle = 0;
        } else if (rot == 1) {
            angle = 45;
        } else {
            angle = -45;
        }
        auto bb = BB{col, row, query_cols - 1, query_rows - 1};
        if (!bb.rotate(angle, data_rows, data_cols)) {
            continue;
        }
        auto temp = get_prefix_sum(bb, data_rows, data_cols, prefixsum_cu);
        float ps = temp / ((float) 3 * (bb.h + 1) * (bb.w + 1));
        graydiff_cu[row * data_cols * 3 + col * 3 + rot] = abs(ps - graysum_avg);
        if (abs(ps - graysum_avg) >= th2){
            continue;
        }
        int start = 0, end = query_rows * query_cols;
        float rmsd = 0.0f;
        for(auto query_idx = start; query_idx < end; query_idx++){
            int curr_col = query_idx % query_cols;
            int curr_row = query_idx / query_cols;
            Point rotated_point;
            if (rot == 0) {
                rotated_point.x = col + curr_col;
                rotated_point.y = row - (query_rows - 1) + curr_row;
            } else if (rot == 1) {
                rotated_point.x = col + query_rotated_1_cu[curr_row * query_cols * 2 + curr_col * 2];
                rotated_point.y = row + query_rotated_1_cu[curr_row * query_cols * 2 + curr_col * 2 + 1];
            } else {
                rotated_point.x = col + query_rotated_2_cu[curr_row * query_cols * 2 + curr_col * 2];
                rotated_point.y = row + query_rotated_2_cu[curr_row * query_cols * 2 + curr_col * 2 + 1];
            }
            
            for(auto ch = 0; ch < 3; ch++){
                query_px[ch] = query_cu[query_idx * 3 + ch];
            }
            if (angle != 0) {
                bilinear_interpolate(rotated_point, 0, data_rows, data_cols, data_cu, data_px);
            } else {
                get_value(data_cu, rotated_point.y, rotated_point.x, 0, data_rows, data_cols, data_px);
            }
            rmsd += ((data_px[0] - query_px[0]) * (data_px[0] - query_px[0])) + ((data_px[1] - query_px[1]) * (data_px[1] - query_px[1])) + ((data_px[2] - query_px[2]) * (data_px[2] - query_px[2]));
        }
        result_cu[row * data_cols * 3 + col * 3 + rot] = sqrt(rmsd / (query_rows * query_cols * 3));
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
    th1 = std::stod(argv[3]);
    th2 = std::stod(argv[4]);
    n = std::stoi(argv[5]);
    auto data_mat = imread(data_img);
    auto query_mat = imread(query_img);
    T<int, int, int> data_sz = data_mat->shape(), query_sz = query_mat->shape();
    int data_rows = std::get<0>(data_sz), data_cols = std::get<1>(data_sz), data_dim = std::get<2>(data_sz);
    int query_rows = std::get<0>(query_sz), query_cols = std::get<1>(query_sz), query_dim = std::get<2>(query_sz);

    std::cout << "Data Image: " << data_rows << " " << data_cols << " " << data_dim << std::endl;
    std::cout << "Query Image: " << query_rows << " " << query_cols << " " << query_dim << std::endl;

    int data_mem = data_rows * data_cols * data_dim, query_mem = query_rows * query_cols * query_dim;

    auto prefixsum_mat = prefixsum(data_mat);
    auto graysum_avg = graysum(query_mat);

    V<float> result_arr(data_mem, std::numeric_limits<float>::max());
    V<float> graydiff_arr(data_mem, std::numeric_limits<float>::max());
    V<float> query_rotated_1_arr(query_cols * query_rows * 2), query_rotated_2_arr(query_cols * query_rows * 2);

    rotate_matrix(query_rows, query_cols, 45, query_rotated_1_arr.data());
    rotate_matrix(query_rows, query_cols, -45, query_rotated_2_arr.data());

    int *data_cu, *query_cu;
    float *result_cu;
    ll *prefixsum_cu;
    float *graydiff_cu;
    float *query_rotated_1_cu, *query_rotated_2_cu;

    cudaMalloc(&data_cu, data_mem * sizeof(int)); 
    cudaMalloc(&query_cu, query_mem * sizeof(int));
    cudaMalloc(&prefixsum_cu, data_rows * data_cols * sizeof(ll));
    cudaMalloc(&result_cu, data_rows * data_cols * 3 * sizeof(float));
    cudaMalloc(&graydiff_cu, data_rows * data_cols * 3 * sizeof(float));
    cudaMalloc(&query_rotated_1_cu, query_rotated_1_arr.size() * sizeof(float));
    cudaMalloc(&query_rotated_2_cu, query_rotated_2_arr.size() * sizeof(float));

    cudaMemcpy(data_cu, data_mat->get(), data_mem * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(query_cu, query_mat->get(), query_mem * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(prefixsum_cu, prefixsum_mat->get(), data_rows * data_cols * sizeof(ll), cudaMemcpyHostToDevice);
    cudaMemcpy(result_cu, result_arr.data(), data_rows * data_cols * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(query_rotated_1_cu, query_rotated_1_arr.data(), query_rows * query_cols * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(query_rotated_2_cu, query_rotated_2_arr.data(), query_rows * query_cols * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(graydiff_cu, graydiff_arr.data(), data_rows * data_cols * 3 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_dim = dim3(std::min(65536, (data_rows * data_cols) / 1024), 3);
    dim3 block_dim = dim3(std::min(128, query_rows * query_cols));
    
    // invoke kernel
    find<<<grid_dim, block_dim>>>(th2, query_cu, query_rows, query_cols, graysum_avg, data_cu, data_rows, data_cols, prefixsum_cu, result_cu, graydiff_cu, query_rotated_1_cu, query_rotated_2_cu);
    cudaDeviceSynchronize();

    cudaMemcpy(result_arr.data(), result_cu, data_mem * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(graydiff_arr.data(), graydiff_cu, data_mem * sizeof(float), cudaMemcpyDeviceToHost);
    
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
    
    V<std::tuple<int, int, int, float, float>> output(n);
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
        output[index] = std::tuple<int, int, int, float, float>(data_rows - 1 - row, col, rot, temp.second, graydiff_arr[temp.first]);
        index --;
    }
    for (int i = 0; i < output_sz; i ++) {
        fout << std::get<3>(output[i]) << " " << std::get<4>(output[i]) << " " << std::get<0>(output[i]) << " " << std::get<1>(output[i]) << " " << std::get<2>(output[i]) << std::endl;
    }
    fout.close();
    cudaFree(result_cu);
    cudaFree(prefixsum_cu);
    cudaFree(query_cu);
    cudaFree(data_cu);
}
