
#include "types.hpp"
V<P<int, double>> QueryHNSW(double *user, int ep, V<int> &indptr, V<int> &index,
                            V<int> &level_offset, int max_level, double *vect,
                            int K, int start, int end, int num_lines, int size,
                            int embedding_size, MPI_Datatype &vector_t,
                            MPI_Win &win);