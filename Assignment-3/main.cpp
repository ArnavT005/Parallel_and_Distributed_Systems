#include <bits/stdc++.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include "hnsw.hpp"
#include "util.hpp"

int main(int argc, char **argv) {
  std::string out_dir = argv[1];
  int K = std::stoi(argv[2]);
  std::string in_file = argv[3];
  std::string out_file = argv[4];
  int max_level, ep;
  V<int> level, index, indptr, level_offset;
  int size, rank;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_File fout;
  //open file to write data to
  if (MPI_File_open(MPI_COMM_WORLD, out_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout) != MPI_SUCCESS) {
    printf("Unable to open file: %s", out_file.c_str());
    return -1;
  }
  MPI_File_close(&fout);
  MPI_Barrier(MPI_COMM_WORLD);
  //read data
  read_int(out_dir + "/max_level.txt", max_level);
  read_int(out_dir + "/ep.txt", max_level);
  read_vect(out_dir + "/level.bin", level);
  read_vect(out_dir + "/index.bin", index);
  read_vect(out_dir + "/indptr.bin", indptr);
  read_vect(out_dir + "/level_offset.bin", level_offset);
  //read embeddings
  uint embedding_size, num_users, num_news;
  get_embedding_info(out_dir + "/vect.bin", embedding_size, num_news);
  get_embedding_info(out_dir + "/user.bin", embedding_size, num_users);
  //create datatype
  MPI_Datatype vector_t;
  MPI_Type_contiguous(embedding_size, MPI_DOUBLE, &vector_t);
  MPI_Type_commit(&vector_t);
  double *vect_buff, *user_buff;
  auto begin = std::chrono::high_resolution_clock::now();
  vect_buff = read_embeddings(out_dir + "/vect.bin", rank, size, vector_t, num_news, embedding_size);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
  printf("Read vect embedding time: %ld\n", elapsed.count());
  begin = std::chrono::high_resolution_clock::now();
  user_buff = read_embeddings(out_dir + "/user.bin", rank, size, vector_t, num_users, embedding_size);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
  printf("Read user embedding time: %ld\n", elapsed.count());
  uint start_news = rank * num_news / size, end_news = (rank + 1) * num_news / size, start_user = rank * num_users / size, end_user = (rank + 1) * num_users / size;
  V<int> counts, displacements;
  int sum = 0;
  for (int i = 0; i < size; i ++) {
    int start_item = i * num_news / size, end_item = (i + 1) * num_news / size;
    counts.push_back(end_item - start_item);
    displacements.push_back(sum);
    sum += (end_item - start_item);
  }
  double* vect = new double[num_news];
  MPI_Allgatherv((void*) vect_buff, end_news - start_news, vector_t, vect, counts.data(), displacements.data(), vector_t, MPI_COMM_WORLD);
  //divide work among threads
  MPI_Datatype items_t;
  MPI_Type_contiguous(K, MPI_UNSIGNED, &items_t);
  MPI_Type_commit(&items_t);
  MPI_File_open(MPI_COMM_WORLD, out_file.c_str(), MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
  MPI_File_seek(fout, start_user * K * sizeof(int), MPI_SEEK_SET);
  V<int> items(K, -1);
  #pragma omp parallel for ordered
  for (int i = start_user; i < end_user; i ++) {
    V<P<int, double>> topk = QueryHNSW(&user_buff[(i - start_user) * ((ll) embedding_size)], ep, indptr, index, level_offset, max_level, vect, K, num_news, embedding_size);
    for (int j = K; j > 0; j --) {
      std::pop_heap(topk.begin(), topk.begin() + j, comparePairMax());
      items[K - j] = topk[j - 1].first;
    }
    #pragma omp ordered
    MPI_File_write(fout, (void*) items.data(), 1, items_t, MPI_STATUS_IGNORE);
  }
  MPI_File_close(&fout);
//   MPI_Win window;
//   std::cout << "Rank and Size: " << rank << " " << size << std::endl;
  
//   MPI_Win_create((void *)vect_buff, embedding_size * (end_news - start_news),
//                  embedding_size * sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD,
//                  &window);
//   MPI_Win_fence(0, window);
// #pragma omp parallel for
//   {
//     for (int i = start_user; i < end_user; i++) {
//       // use double buffer in place of vectors (user, vect)
//       V<P<int, double>> topk =
//           QueryHNSW(user_buff, ep, indptr, index, level_offset, max_level,
//                     vect_buff, K, start_news, end_news, num_news, size,
//                     embedding_size, vector_t, window);
//     }
//   }
//   MPI_Win_free(&window);
  MPI_Finalize();
  delete vect;
  delete vect_buff;
  delete user_buff;
}
