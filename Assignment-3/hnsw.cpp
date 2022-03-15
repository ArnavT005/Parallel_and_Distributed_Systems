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
  return dist / (norm(a, sz) * norm(b, sz));
}

// double *get_vec(double *vect, int node, int start, int end, int num_lines, int size, int embedding_size, MPI_Datatype &vector_t, MPI_Win &win) {
//   if (node >= start && node < end) {
//     return &vect[(node - start) * embedding_size];
//   } else {
//     double *v = new double[embedding_size];
//     int target;
//     for (int i = 0; i <= size; i++) {
//       if (i * num_lines / size > node) {
//         target = (i - 1);
//       }
//     }
//     printf("Fetching from target: %d\n", target);
//     printf("Fetching from index: %d\n", node - target * num_lines / size);
//     MPI_Win_fence(0, win);
//     MPI_Get((void *)v, 1, vector_t, target, node - target * num_lines / size, 1,
//             vector_t, win);
//     MPI_Win_fence(0, win);
//     printf("Fetching complete!\n");
//     return v;
//   }
// }

// V<P<int, double>> SearchLayer(double *user, V<P<int, double>> &candidates,
//                               V<int> &indptr, V<int> &index,
//                               V<int> &level_offset, int level, V<bool> &visited,
//                               double *vect, int K, int start_news, int end_news,
//                               int num_lines, int size, int embedding_size,
//                               MPI_Datatype &vector_t, MPI_Win &win) {
//   V<P<int, double>> topk = candidates;
//   while (candidates.size() > 0) {
//     std::pop_heap(candidates.begin(), candidates.end(), comparePairMax());
//     int ep = candidates[candidates.size() - 1].first;
//     candidates.pop_back();
//     int start = indptr[ep] + level_offset[level],
//         end = indptr[ep] + level_offset[level + 1];
//     for (int i = start; i < end; i++) {
//       if (visited[index[i]] || index[i] == -1) {
//         continue;
//       }
//       visited[index[i]] = true;
//       double *ep_vec = get_vec(vect, index[i], start_news, end_news, num_lines,
//                                size, embedding_size, vector_t, win);
//       printf("Fetched embedding for %d\n", index[i]);
//       double _dist = cosine_dist(user, ep_vec, embedding_size);
//       if (_dist > topk[0].first && topk.size() == K) {
//         continue;
//       }
//       topk.push_back(P<int, double>(index[i], _dist));
//       std::push_heap(topk.begin(), topk.end(), comparePairMax());
//       if (topk.size() > K) {
//         std::pop_heap(topk.begin(), topk.end(), comparePairMax());
//         topk.pop_back();
//       }
//       candidates.push_back(P<int, double>(index[i], _dist));
//       std::push_heap(candidates.begin(), candidates.end(), comparePairMax());
//     }
//   }
//   return topk;
// }
V<P<int, double>> SearchLayer(double *user, V<P<int, double>> &candidates, V<int> &indptr, V<int> &index, V<int> &level_offset, int level, V<bool> &visited, double *vect, int K, int num_lines, int embedding_size) {
  V<P<int, double>> topk = candidates;
  while (candidates.size() > 0) {
    std::pop_heap(candidates.begin(), candidates.end(), comparePairMax());
    int ep = candidates[candidates.size() - 1].first;
    candidates.pop_back();
    int start = indptr[ep] + level_offset[level], end = indptr[ep] + level_offset[level + 1];
    for (int i = start; i < end; i++) {
      if (visited[index[i]] || index[i] == -1) {
        continue;
      }
      visited[index[i]] = true;
      printf("Fetched embedding for %d\n", index[i]);
      double _dist = cosine_dist(user, &vect[index[i] * embedding_size], embedding_size);
      if (_dist > topk[0].first && topk.size() == K) {
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

// V<P<int, double>> QueryHNSW(double *user, int ep, V<int> &indptr, V<int> &index, V<int> &level_offset, int max_level, double *vect, int K, int start, int end, int num_lines, int size, int embedding_size, MPI_Datatype &vector_t, MPI_Win &win) {
//   V<P<int, double>> topk;
//   auto ep_vec = get_vec(vect, ep, start, end, num_lines, size, embedding_size, vector_t, win);
//   topk.push_back(P<int, double>(ep, cosine_dist(user, ep_vec, embedding_size)));
//   V<bool> visited(num_lines, false);
//   visited[ep] = true;
//   for (int level = max_level; level >= 0; level--) {
//     // printf("Calling search layer: %d\n", level);
//     topk = SearchLayer(user, topk, indptr, index, level_offset, level, visited,
//                        vect, K, start, end, num_lines, size, embedding_size,
//                        vector_t, win);
//   }
//   return topk;
// }
V<P<int, double>> QueryHNSW(double *user, int ep, V<int> &indptr, V<int> &index, V<int> &level_offset, int max_level, double *vect, int K, int num_lines, int embedding_size) {
  V<P<int, double>> topk;
  topk.push_back(P<int, double>(ep, cosine_dist(user, &vect[ep * embedding_size], embedding_size)));
  V<bool> visited(num_lines, false);
  visited[ep] = true;
  for (int level = max_level; level >= 0; level--) {
    // printf("Calling search layer: %d\n", level);
    topk = SearchLayer(user, topk, indptr, index, level_offset, level, visited, vect, K, num_lines, embedding_size);
  }
  return topk;
}