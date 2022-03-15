#include "types.hpp"

struct comparePairMax {
    bool operator()(const P<int, double> &a, const P<int, double> &b) {
    if (a.second < b.second) {
      return true;
    } else if (a.second == b.second) {
      return b.first < a.first;
    } else {
      return false;
    }
  }
};
// V<P<int, double>> QueryHNSW(double *user, int ep, V<int> &indptr, V<int> &index, V<int> &level_offset, int max_level, double *vect, int K, int start, int end, int num_lines, int size, int embedding_size, MPI_Datatype &vector_t, MPI_Win &win);
V<P<int, double>> QueryHNSW(double *user, int ep, V<int> &indptr, V<int> &index, V<int> &level_offset, int max_level, double *vect, int K, int num_lines, int embedding_size);