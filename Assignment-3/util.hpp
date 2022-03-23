#pragma once
#include "types.hpp"

void read_int(std::string file_name, int &res);
void read_vect(std::string file_name, V<int> &vect);
void get_embedding_info(std::string file_name, uint &embedding_size,
                        uint &num_lines);
double *read_embeddings(std::string file_name, int rank, int size,
                        MPI_Datatype &vector_t, uint num_lines,
                        uint embedding_size);
void convert_to_txt(std::string in_file, std::string out_file);
void make_counts_displ(int total_count, int size, V<int> &counts,
                       V<int> &displacements);
void write_txt_embeddings(std::string filepath, int *items, int num_items,
                          int item_size);
void read_embedding_txt(std::string file_name, V<double> &vect);
