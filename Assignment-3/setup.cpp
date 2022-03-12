#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char **argv) {
    std::string in_dir = argv[1], out_dir = argv[2];
    std::ifstream fin(in_dir + "/users.txt", std::ios::in);
    std::ofstream fout(out_dir + "/users.bin", std::ios::out | std::ios::binary);
    std::string line;
    while (std::getline(fin, line)) {
        std::stringstream stream(line);
        double temp;
        while (stream >> temp) {
            fout.write((char*) &temp, sizeof(double));
        }
    }
    fin.close();
    fout.close();
}