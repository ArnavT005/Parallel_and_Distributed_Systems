#include "util.hpp"

void read_int(std::string file_name, int &res) {
  std::ifstream fin(file_name, std::ios::in);
  fin >> res;
  fin.close();
  printf("Read int from %s : %d\n", file_name.c_str(), res);
}

void read_vect(std::string file_name, V<int> &vect) {
  MPI_File fin;
  if (MPI_File_open(MPI_COMM_WORLD, file_name.c_str(), MPI_MODE_RDONLY,
                    MPI_INFO_NULL, &fin) != MPI_SUCCESS) {
    printf("Unable to open file: %s", file_name.c_str());
    return;
  }
  MPI_File_seek(fin, 0, MPI_SEEK_SET);
  uint num_elements;
  MPI_File_read(fin, &num_elements, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
  vect.resize(num_elements);
  MPI_Datatype vect_t;
  MPI_Type_contiguous(num_elements, MPI_UNSIGNED, &vect_t);
  MPI_Type_commit(&vect_t);
  MPI_File_read(fin, vect.data(), 1, vect_t, MPI_STATUS_IGNORE);
  MPI_File_close(&fin);
  printf("Read vector from %s, got %ld elements\n", file_name.c_str(),
         vect.size());
}

void get_embedding_info(std::string file_name, uint &embedding_size,
                        uint &num_lines) {
  MPI_File fin;
  if (MPI_File_open(MPI_COMM_WORLD, file_name.c_str(), MPI_MODE_RDONLY,
                    MPI_INFO_NULL, &fin) != MPI_SUCCESS) {
    printf("Unable to open file: %s", file_name.c_str());
    return;
  }
  MPI_File_seek(fin, 0, MPI_SEEK_SET);
  MPI_File_read(fin, &num_lines, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
  MPI_File_read(fin, &embedding_size, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
  MPI_File_close(&fin);
}

double *read_embeddings(std::string file_name, int rank, int size,
                        MPI_Datatype &vector_t, uint num_lines,
                        uint embedding_size) {
  MPI_File fin;
  if (MPI_File_open(MPI_COMM_WORLD, file_name.c_str(), MPI_MODE_RDONLY,
                    MPI_INFO_NULL, &fin) != MPI_SUCCESS) {
    printf("Unable to open file: %s", file_name.c_str());
    return nullptr;
  }
  int start = rank * num_lines / size;
  int end = (rank + 1) * num_lines / size;
  printf("Starting to read %d-%d lines, with embedding size: %d\n", start, end,
         embedding_size);
  MPI_File_seek(fin, 2 * sizeof(uint) + start * embedding_size * sizeof(double),
                MPI_SEEK_SET);
  double *buff = new double[((ll)embedding_size) * (end - start)];
  for (int i = start; i < end; i++) {
    MPI_File_read(fin, &buff[(i - start) * embedding_size], 1, vector_t,
                  MPI_STATUS_IGNORE);
  }
  MPI_File_close(&fin);
  return buff;
}

void convert_to_txt(std::string in_file, std::string out_file) {
  std::ifstream fin(in_file, std::ios::in | std::ios::binary);
  std::ofstream fout(out_file, std::ios::out);
  if (!fout) {
    std::cerr << "Cannot open file" << std::endl;
    return;
  }
  unsigned int line_num = 0, embedding_size = 0;
  fin.seekg(0, std::ios::beg);
  fin.read((char *)&line_num, sizeof(line_num));
  fin.read((char *)&embedding_size, sizeof(embedding_size));
  printf("Reading %d lines, with embedding_size: %d\n", line_num,
         embedding_size);
  std::string line;
  for (unsigned int i = 0; i < line_num; i++) {
    for (unsigned int j = 0; j < embedding_size; j++) {
      uint temp;
      fin.read((char *)&temp, sizeof(uint));
      // printf("%d ", temp);
      fout << temp << " ";
    }
    // printf("\n");
    fout << std::endl;
  }
  fin.close();
  fout.close();
}
