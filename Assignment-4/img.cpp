#include "img.hpp"

template <typename K>
matrix<K>::matrix() {
    mat.clear();
    row = col = dim = 0;
}

template <typename K>
matrix<K>::matrix(int R, int C, int k) {
    mat.resize(R);
    for (int i = 0; i < R; i ++) {
        mat[i].resize(C);
        for (int j = 0; j < C; j ++) {
            mat[i][j].resize(k);
        }
    }
    row = R;
    col = C;
    dim = k;
}

template <typename K>
V<V<K>> matrix<K>::operator[](int index) {
    return mat[index];
}

template <typename K>
void matrix<K>::resize(int row, int col, int dim, bool init, K val) {
    mat.resize(row);
    for (int i = 0; i < row; i ++) {
        mat[i].resize(col);
        for (int j = 0; j < col; j ++) {
            mat[i][j].resize(dim);
            if (init) {
                for (int k = 0; k < dim; k ++) {
                    mat[i][j][k] = val;
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

template <typename T>
void matrix<T>::read_img(std::string file_img) {
    std::string line;
    std::ifstream fin(file_img, std::ios::in);
    std::getline(fin, line);
    std::stringstream stream(line);
    int col, row;
    stream >> col >> row;
    resize(row, col, 3);
    std::getline(fin, line);
    stream.str("");
    stream.clear();
    stream << line;
    std::cout << row << " " << col << std::endl;
    for (int i = 0; i < row; i ++) {
        for (int j = 0; j < col; j ++) {
            int R, G, B;
            stream >> R >> G >> B;
            mat[i][j][0] = R;
            mat[i][j][1] = G;
            mat[i][j][2] = B;
        }
    }
}
