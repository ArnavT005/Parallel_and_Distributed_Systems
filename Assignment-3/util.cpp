
#include "util.hpp"

#include "types.hpp"
void get_embedding_info(std::string file_name, unsigned int &embedding_size,
                        unsigned int &num_lines) {
  MPI_File fin;
  if (MPI_File_open(MPI_COMM_WORLD, file_name.c_str(), MPI_MODE_RDONLY,
                    MPI_INFO_NULL, &fin) != MPI_SUCCESS) {
    printf("Unable to open file: %s", file_name.c_str());
  }
  MPI_File_seek(fin, 0, MPI_SEEK_SET);
  MPI_File_read(fin, &num_lines, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
  MPI_File_read(fin, &embedding_size, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
  MPI_File_close(&fin);
}
double *read_embeddings(std::string file_name, V<V<double>> &vect, int rank,
                        int size, MPI_Datatype &vector_t,
                        unsigned int num_lines, unsigned int embedding_size) {
  MPI_File fin;
  if (MPI_File_open(MPI_COMM_WORLD, file_name.c_str(), MPI_MODE_RDONLY,
                    MPI_INFO_NULL, &fin) != MPI_SUCCESS) {
    printf("Unable to open file: %s", file_name.c_str());
  }
  int start = rank * num_lines / size;
  int end = (rank + 1) * num_lines / size;
  printf("Starting to read %d-%d lines, with embedding size: %d\n", start, end,
         embedding_size);

  MPI_File_seek(
      fin, 2 * sizeof(unsigned int) + start * embedding_size * sizeof(double),
      MPI_SEEK_SET);
  // MPI_File_set_view(fin, 1 + start * embedding_size, MPI_DOUBLE, vector_t,
  //                   "native", MPI_INFO_NULL);
  // vect.resize(embedding_size);
  auto buff = new double[(long long)(embedding_size) * (end - start)];

  // vect.resize(end - start, V<double>(embedding_size));
  for (int i = start; i < end; i++) {
    // MPI_File_read(fin, vect[i - start].data(), 1, vector_t,
    // MPI_STATUS_IGNORE);
    MPI_File_read(fin, &buff[(i - start) * embedding_size], 1, vector_t,
                  MPI_STATUS_IGNORE);
    // MPI_File_read(fin, &buff[0], end - start, vector_t, MPI_STATUS_IGNORE);

    // std::cout << "read " << i << std::endl;
  }
  MPI_File_close(&fin);
  return buff;
}

void read_vect(std::string file_name, V<int> &vect) {
  MPI_File fin;
  if (MPI_File_open(MPI_COMM_WORLD, file_name.c_str(), MPI_MODE_RDONLY,
                    MPI_INFO_NULL, &fin) != MPI_SUCCESS) {
    printf("Unable to open file: %s", file_name.c_str());
  }
  MPI_File_seek(fin, 0, MPI_SEEK_SET);
  unsigned int num_elements;
  MPI_File_read(fin, &num_elements, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
  vect.resize(num_elements);
  MPI_Datatype vect_t;
  MPI_Type_contiguous(num_elements, MPI_UNSIGNED, &vect_t);
  MPI_Type_commit(&vect_t);
  MPI_File_read(fin, vect.data(), 1, vect_t, MPI_STATUS_IGNORE);
  printf("Read vector from %s, got %d elements\n", file_name.c_str(),
         vect.size());
  // for (int i = 0; i < 5; i++) printf("%d ", vect[i]);
  // printf("\n");
}

void read_int(std::string file_name, int &res) {
  std::ifstream fin(file_name, std::ios::in);
  fin >> res;
  fin.close();
  printf("Read int from %s : %d\n", file_name.c_str(), res);
}