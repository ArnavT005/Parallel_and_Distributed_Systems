#include "hnsw.hpp"

double norm(double *a, int sz) {
  double norm = 0;
  for (int i = 0; i < sz; i++) {
    norm += a[i] * a[i];
  }
  return sqrt(norm);
}

double cosine_dist(double *a, double *b, int sz) {
  double dist = 0;
  for (int i = 0; i < sz; i++) {
    dist += (a[i] * b[i]);
  }
  return 1.0 - (dist / (norm(a, sz) * norm(b, sz)));
}

V<P<int, double>> SearchLayer(double *user, V<P<int, double>> &candidates,
                              V<int> &indptr, V<int> &index,
                              V<int> &level_offset, int level, V<bool> &visited,
                              double *vect, int K, int num_lines,
                              int embedding_size) {
  V<P<int, double>> topk = candidates;
  while (candidates.size() > 0) {
    std::pop_heap(candidates.begin(), candidates.end(), comparePairMax());
    int ep = candidates[candidates.size() - 1].first;
    candidates.pop_back();
    int start = indptr[ep] + level_offset[level],
        end = indptr[ep] + level_offset[level + 1];
    for (int i = start; i < end; i++) {
      if (index[i] == -1 || visited[index[i]]) {
        continue;
      }
      visited[index[i]] = true;
      double _dist =
          cosine_dist(user, &vect[index[i] * embedding_size], embedding_size);
      if (_dist > topk[0].second && topk.size() == K) {
        continue;
      }
      topk.push_back(P<int, double>(index[i], _dist));
      std::push_heap(topk.begin(), topk.end(), comparePairMax());
      if (topk.size() > K) {
        std::pop_heap(topk.begin(), topk.end(), comparePairMax());
        topk.pop_back();
      }
      candidates.push_back(P<int, double>(index[i], _dist));
      std::push_heap(candidates.begin(), candidates.end(), comparePairMax());
    }
  }
  return topk;
}

V<P<int, double>> QueryHNSW(double *user, int ep, V<int> &indptr, V<int> &index,
                            V<int> &level_offset, int max_level, double *vect,
                            int K, int num_lines, int embedding_size) {
  V<P<int, double>> topk;
  topk.push_back(P<int, double>(
      ep, cosine_dist(user, &vect[ep * embedding_size], embedding_size)));
  V<bool> visited(num_lines, false);
  visited[ep] = true;
  int thread_id = omp_get_thread_num();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  for (int level = max_level; level >= 0; level--) {
    printf("[%d@%d] Calling search layer %d\n", thread_id, rank, level);
    topk = SearchLayer(user, topk, indptr, index, level_offset, level, visited,
                       vect, K, num_lines, embedding_size);
  }
  return topk;
}