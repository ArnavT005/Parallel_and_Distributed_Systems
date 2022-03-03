#include <fstream>
#include <iostream>
#include <string>

typedef unsigned char uchar;

bool is_little_endian() {
    int test_num = 1;
    char test_ch = *((char *) &test_num);
    if (test_ch == 1) {
        return true;
    } else {
        return false;
    }
}

void write_int(std::ofstream *fout, int *val) {
    char num[4];
    for (int i = 3; i > -1; i --) {
        num[i] = *(((char*) val) + i);
        fout->write(&num[i], sizeof(char));
    }
}

int main(int argc, char** argv) {
    std::ifstream fin("input.txt", std::ios::in);
    std::ofstream fout("input.dat", std::ios::out | std::ios::binary);
    int num_edges = std::stoi(argv[1]);
    for (int i = 0; i < num_edges; i ++) {
        int a, b;
        fin >> a >> b;
        if (!is_little_endian()) {
            fout.write((char*) &a, sizeof(a));
            fout.write((char*) &b, sizeof(b));
        } else {
            write_int(&fout, &a);
            write_int(&fout, &b);
        }
    }
    fin.close();
    fout.close();
}