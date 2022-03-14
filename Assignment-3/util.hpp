#pragma once

#include "types.hpp"

void get_embedding_info(std::string file_name, unsigned int &embedding_size,
                        unsigned int &num_lines);
double *read_embeddings(std::string file_name, V<V<double>> &vect, int rank,
                        int size, MPI_Datatype &vector_t,
                        unsigned int num_lines, unsigned int embedding_size);
void read_vect(std::string file_name, V<int> &vect);
void read_int(std::string file_name, int &res);