#include "img.hpp"


matrix::matrix() {
    mat.clear();
    row = col = dim = 0;
}

matrix::matrix(int R, int C, int K) {
    mat.resize(R);
    for (int i = 0; i < R; i ++) {
        mat[i].resize(C);
        for (int j = 0; j < C; j ++) {
            mat[i][j].resize(K);
            for (int k = 0; k < K; k ++) {
                mat[i][j][k] = 0;
            }
        }
    }
    row = R;
    col = C;
    dim = K;
}

V<V<int>> matrix::operator[](int index) {
    return mat[index];
}

void matrix::resize(int row, int col, int dim, bool init, int val) {
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

T<int, int, int> matrix::shape() {
    return std::make_tuple(row, col, dim);
}

void read_img(std::string file_img, matrix &mat) {
    std::string line;
    std::ifstream fin(file_img, std::ios::in);
    std::getline(fin, line);
    std::stringstream stream(line);
    int col, row;
    stream >> col >> row;
    mat.resize(row, col, 3);
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
