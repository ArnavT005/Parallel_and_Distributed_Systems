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

void read_int(std::ifstream *fin, int *val) {
    uchar num[4];
    for (int i = 0; i < 4; i ++) {
        fin->read((char*) &num[i], sizeof(char));
    }
    *val = ((((int) num[0]) << 24) | (((int) num[1]) << 16) | (((int) num[2]) << 8) | ((int) num[3]));
}

int main(int argc, char** argv) {
    std::ifstream fin("output.dat", std::ios::in);
    std::ofstream fout("output.txt");
    int num_nodes = std::stoi(argv[1]);
    int num_rec = std::stoi(argv[2]);
    for (int i = 0; i < num_nodes; i ++) {
        int degree, id, score;
        if (!is_little_endian()) {
            fin.read((char*) &degree, sizeof(degree));
        } else {
            read_int(&fin, &degree);
        }
        fout << i << " " << degree << ",";
        for (int j = 0; j < num_rec; j ++) {
            if (!is_little_endian()) {
                fin.read((char*) &id, sizeof(id));
                fin.read((char*) &score, sizeof(score));
            } else {
                read_int(&fin, &id);
                read_int(&fin, &score);
            }
            fout << id << " " << score << ",";
        }
        fout << "\n";
    }
    fin.close();
    fout.close();
    int num[4];
    num[0] = 'N'; num[1] = 'U'; num[2] = num[3] = 'L';
    if (!is_little_endian()) {
        int null = ((((int) num[3]) << 24) | (((int) num[2]) << 16) | (((int) num[1]) << 8) | ((int) num[0]));
        std::cout << "Null value: " <<  null << std::endl;
    } else {
        int null = ((((int) num[0]) << 24) | (((int) num[1]) << 16) | (((int) num[2]) << 8) | ((int) num[3]));
        std::cout << "Null value: " <<  null << std::endl;
    }
}