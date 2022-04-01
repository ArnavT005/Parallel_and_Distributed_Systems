#include "img.hpp"

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

unique_ptr<matrix<int>> imread(string file_img) {
    std::string line;
    std::ifstream fin(file_img, std::ios::in);
    std::getline(fin, line);
    std::stringstream stream(line);
    int row, col;
    stream >> row >> col;
    auto img_mat = make_unique<matrix<int>>(row, col, 3);
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
    return move(img_mat);
}

unique_ptr<matrix<float>> rgb2gray(matrix<int>* img){
    auto shape = img->shape();
    auto row = std::get<0>(shape);
    auto col = std::get<1>(shape);
    auto dim = std::get<2>(shape);

    auto gray_img = make_unique<matrix<float>>(row, col, 1);
    for (int i = 0; i < row; i ++) {
        for (int j = 0; j < col; j ++) {
            int R = img->get(i, j, 0);
            int G = img->get(i, j, 1);
            int B = img->get(i, j, 2);
            float gray = (R + G + B) / 3.0f;
            gray_img->set(i, j, 0, gray);
        }
    }
    return move(gray_img);
}

float graysum(matrix<int>* img){
    auto shape = img->shape();
    auto row = std::get<0>(shape);
    auto col = std::get<1>(shape);
    auto dim = std::get<2>(shape);
    float sum = 0;
    for (int i = 0; i < row; i ++) {
        for (int j = 0; j < col; j ++) {
            float dim_sum = 0;
            for(int k = 0; k < dim; k++){
                dim_sum += img->get(i,j,k);
            }
            sum += dim_sum / dim;
        }
    }
    sum /= (row * col);
    return sum;
}


unique_ptr<matrix<float>> prefixsum(matrix<float>* img){
    auto shape = img->shape();
    auto row = std::get<0>(shape);
    auto col = std::get<1>(shape);
    auto dim = std::get<2>(shape);

    assert(dim == 1);

    auto ps = make_unique<matrix<float>>();
    ps->resize(row, col, 1, true, 0.0f);
    for (int i = 0; i < row; i ++) {
        for (int j = 0; j < col; j ++) {
            int val = img->get(i, j, 0);
            float sum = 0;
            if (i != 0 && j != 0){
                sum = val + ps->get(i-1, j, 0) + ps->get(i, j-1, 0) - ps->get(i-1,j-1,0);
            } else if (i != 0){
                sum = val + ps->get(i-1, j, 0);
            } else if (j != 0){
                sum = val + ps->get(i, j-1, 0);
            } else {
                sum = float(val);
            }
            ps->set(i, j, 0, sum);
        }
    }
    return move(ps);
}

