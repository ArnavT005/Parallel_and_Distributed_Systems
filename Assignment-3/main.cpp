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
  V<V<double>> user_embeddings, news_embeddings;
  int size, rank;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  read_int(out_dir + "/max_level.txt", max_level);
  read_int(out_dir + "/ep.txt", max_level);
  read_vect(out_dir + "/level.bin", level);
  read_vect(out_dir + "/index.bin", index);
  read_vect(out_dir + "/indptr.bin", indptr);
  read_vect(out_dir + "/level_offset.bin", level_offset);

  unsigned int embedding_size, num_users, num_news;
  get_embedding_info(out_dir + "/vect.bin", embedding_size, num_news);
  get_embedding_info(out_dir + "/user.bin", embedding_size, num_users);
  std::cout << embedding_size << " " << num_users << " " << num_news
            << std::endl;
  MPI_Datatype vector_t;
  MPI_Type_contiguous(embedding_size, MPI_DOUBLE, &vector_t);
  MPI_Type_commit(&vector_t);
  V<V<double>> vect, user;
  double *vect_buff, *user_buff;
  auto begin = std::chrono::high_resolution_clock::now();
  vect_buff = read_embeddings(out_dir + "/vect.bin", vect, rank, size, vector_t,
                              num_news, embedding_size);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
  printf("Read vect embedding time: %ld\n", elapsed.count());

  // std::cout << vect[num_news - 1][0] << " " << vect[num_news - 1][1] << " "
  //           << vect[num_news - 1][2] << " " << vect[num_news - 1][3] << " "
  //           << vect[num_news - 1][4] << std::endl;
  // std::cout << std::endl;
  begin = std::chrono::high_resolution_clock::now();

  user_buff = read_embeddings(out_dir + "/user.bin", user, rank, size, vector_t,
                              num_users, embedding_size);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
  printf("Read user embedding time: %ld\n", elapsed.count());

  // std::cout << user[5][0] << " " << user[5][1] << " " << user[5][2] << " "
  //           << user[5][3] << " " << user[5][4] << std::endl;
  // std::cout << std::endl;
  MPI_Win window;
  std::cout << "Rank and Size: " << rank << " " << size << std::endl;
  unsigned int start_news = rank * num_news / size,
               end_news = (rank + 1) * num_news / size,
               start_user = rank * num_users / size,
               end_user = (rank + 1) * num_users / size;
  MPI_Win_create((void *)vect_buff, embedding_size * (end_news - start_news),
                 embedding_size * sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD,
                 &window);
  MPI_Win_fence(0, window);
#pragma omp parallel for
  {
    for (int i = start_user; i < end_user; i++) {
      // use double buffer in place of vectors (user, vect)
      V<P<int, double>> topk =
          QueryHNSW(user_buff, ep, indptr, index, level_offset, max_level,
                    vect_buff, K, start_news, end_news, num_news, size,
                    embedding_size, vector_t, window);
    }
  }
  MPI_Win_free(&window);
  MPI_Finalize();
}
