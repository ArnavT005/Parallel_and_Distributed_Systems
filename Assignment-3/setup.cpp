#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

void write_vector_binary(std::string in_dir, std::string out_dir,
                         std::string file_name) {
  std::ifstream fin(in_dir + "/" + file_name + ".txt", std::ios::in);
  std::string line;
  std::ofstream fout(out_dir + "/" + file_name + ".bin",
                     std::ios::out | std::ios::binary | std::ios::ate);
  if (!fout) {
    std::cerr << "Cannot open file" << std::endl;
    return;
  }
  unsigned int c = 0;
  fout.seekp(sizeof(c), std::ios::beg);
  while (std::getline(fin, line)) {
    std::stringstream stream(line);
    int num;
    while (stream >> num) {
      fout.write((char *)&num, sizeof(int));
      c++;
    }
  }
  fin.close();
  fout.seekp(0, std::ios::beg);
  fout.write((char *)&c, sizeof(c));
  fout.close();
  printf("Wrote %d doubles to %s\n", c,
         (out_dir + "/" + file_name + ".bin").c_str());
}

void convert_to_binary(std::string in_dir, std::string out_dir,
                       std::string file_name) {
  std::ifstream fin(in_dir + "/" + file_name + ".txt", std::ios::in);
  std::ofstream fout(out_dir + "/" + file_name + ".bin",
                     std::ios::out | std::ios::binary | std::ios::ate);
  if (!fout) {
    std::cerr << "Cannot open file" << std::endl;
    return;
  }
  std::string line;
  unsigned int line_num = 0, embedding_size = 0;
  fout.seekp(2 * sizeof(line_num), std::ios::beg);
  while (std::getline(fin, line)) {
    std::stringstream stream(line);
    double temp;
    embedding_size = 0;
    while (stream >> temp) {
      fout.write((char *)&temp, sizeof(double));
      embedding_size++;
    }
    line_num++;
  }

  printf("Read %d lines, with embedding_size: %d\n", line_num, embedding_size);
  fin.close();
  fout.seekp(0, std::ios::beg);
  fout.write((char *)&line_num, sizeof(line_num));
  fout.write((char *)&embedding_size, sizeof(embedding_size));
  fout.close();
}

int main(int argc, char **argv) {
  std::string in_dir = argv[1], out_dir = argv[2];
  convert_to_binary(in_dir, out_dir, "user");
  convert_to_binary(in_dir, out_dir, "vect");
  write_vector_binary(in_dir, out_dir, "level");
  write_vector_binary(in_dir, out_dir, "level_offset");
  write_vector_binary(in_dir, out_dir, "index");
  write_vector_binary(in_dir, out_dir, "indptr");
}