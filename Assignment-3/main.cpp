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
  // read data
  read_int(out_dir + "/max_level.txt", max_level);
  read_int(out_dir + "/ep.txt", ep);
  read_vect(out_dir + "/level.bin", level);
  read_vect(out_dir + "/index.bin", index);
  read_vect(out_dir + "/indptr.bin", indptr);
  read_vect(out_dir + "/level_offset.bin", level_offset);
  // read embeddings
  uint embedding_size, num_users, num_news;
  get_embedding_info(out_dir + "/vect.bin", embedding_size, num_news);
  get_embedding_info(out_dir + "/user.bin", embedding_size, num_users);
  // create datatype
  MPI_Datatype vector_t;
  // Datatype is float here
  MPI_Type_contiguous(embedding_size, MPI_FLOAT, &vector_t);
  MPI_Type_commit(&vector_t);
  double *vect_buff, *user_buff;
  auto begin = std::chrono::high_resolution_clock::now();
  vect_buff = read_embeddings(out_dir + "/vect.bin", rank, size, vector_t,
                              num_news, embedding_size);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
  printf("Read vect embedding time: %ld\n", elapsed.count());
  begin = std::chrono::high_resolution_clock::now();
  user_buff = read_embeddings(out_dir + "/user.bin", rank, size, vector_t,
                              num_users, embedding_size);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
  printf("Read user embedding time: %ld\n", elapsed.count());
  uint start_news = rank * num_news / size,
       end_news = (rank + 1) * num_news / size,
       start_user = rank * num_users / size,
       end_user = (rank + 1) * num_users / size;
  V<int> counts, displacements;
  make_counts_displ(num_news, size, counts, displacements);

  double *vect = new double[num_news * embedding_size];
  MPI_Allgatherv((void *)vect_buff, end_news - start_news, vector_t, vect,
                 counts.data(), displacements.data(), vector_t, MPI_COMM_WORLD);
  // divide work among threads
  MPI_Datatype items_t;
  MPI_Type_contiguous(K, MPI_UNSIGNED, &items_t);
  MPI_Type_commit(&items_t);
  int *items = new int[(end_user - start_user) * K];
  auto begin_1 = std::chrono::steady_clock::now();

#pragma omp parallel
  {
#pragma omp single
    {
      for (int i = start_user; i < end_user; i++) {
#pragma omp task
        {
          int thread_id = omp_get_thread_num();

          V<P<int, double>> topk =
              QueryHNSW(&user_buff[(i - start_user) * ((ll)embedding_size)], ep,
                        indptr, index, level_offset, max_level, vect, K,
                        num_news, embedding_size);
          for (int j = K; j > 0; j--) {
            std::pop_heap(topk.begin(), topk.begin() + j, comparePairMax());
            items[(i - start_user) * K + j - 1] = topk[j - 1].first;
          }
          printf("[%d@%d] User %d done\n", thread_id, rank, i);
        }
      }
    }
  }
  auto end_1 = std::chrono::steady_clock::now();
  auto elapsed_1 =
      std::chrono::duration_cast<std::chrono::seconds>(end_1 - begin_1);
  printf("[%d] Processing time: %ld\n", rank, elapsed_1.count());
  int *all_items = nullptr;
  if (rank == 0) {
    all_items = new int[num_users * K];
  }
  counts.clear();
  displacements.clear();
  make_counts_displ(num_users, size, counts, displacements);
  MPI_Gatherv(items, (end_user - start_user), items_t, all_items, counts.data(),
              displacements.data(), items_t, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    write_txt_embeddings(out_file, all_items, num_users, K);
  }
  MPI_Finalize();
  delete items;
  delete vect;
  delete vect_buff;
  delete user_buff;
  if (rank == 0) {
    delete all_items;
  }
}
