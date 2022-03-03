#include <string>
#include <mpi.h>
#include <assert.h>
#include <fstream>
#include <cstring>
#include <math.h>
#include <bits/stdc++.h>
#include "randomizer.hpp"

typedef unsigned char uchar;
typedef unsigned int uint;

namespace misc {
    
    int min(int a, int b) {
        return ((a <= b) ? a : b);
    }

    bool is_little_endian() {
        int test_num = 1;
        char test_ch = *((char *) &test_num);
        if (test_ch == 1) {
            return true;
        } else {
            return false;
        }
    }

    void read_int(std::ifstream *fin, int *val) {
        uchar num[4];
        for (int i = 0; i < 4; i ++) {
            fin->read((char*) &num[i], sizeof(char));
        }
        *val = ((((int) num[0]) << 24) | (((int) num[1]) << 16) | (((int) num[2]) << 8) | ((int) num[3]));
    }

    void write_int(std::ofstream *fout, int *val) {
        char num[4];
        for (int i = 3; i > -1; i --) {
            num[i] = *(((char*) val) + i);
            fout->write(&num[i], sizeof(char));
        }
    }

    struct comparePairMin {
        bool operator() (const std::pair<int, int> &a, const std::pair<int, int> &b) {
            if (a.first > b.first) {
                return true;
            } else if (a.first == b.first) {
                return a.second < b.second;
            } else {
                return false;
            }
        }
    };

    //Notice how the randomizer is being used in this dummy function
    void print_random(int tid, int num_nodes, Randomizer r){
        int this_id = tid;
        int num_steps = 10;
        int num_child = 20;

        std::string s = "Thread " + std::to_string(this_id) + "\n";
        std::cout << s;

        for(int i=0;i<num_nodes;i++){
            //Processing one node
            for(int j=0; j<num_steps; j++){
                if(num_child > 0){
                    //Called only once in each step of random walk using the original node id 
                    //for which we are calculating the recommendations
                    int next_step = r.get_random_value(i);
                    //Random number indicates restart
                    if(next_step<0){
                        std::cout << "Restart \n";
                    }else{
                        //Deciding next step based on the output of randomizer which was already called
                        int child = next_step % 20; //20 is the number of child of the current node
                        std::string s = "Thread " + std::to_string(this_id) + " rand " + std::to_string(child) + "\n";
                        std::cout << s;
                    }
                }else{
                    std::cout << "Restart \n";
                }
            }
        }
    }
}


int main(int argc, char* argv[]){
    assert(argc > 8);
    std::string graph_file = argv[1];
    int num_nodes = std::stoi(argv[2]);
    int num_edges = std::stoi(argv[3]);
    float restart_prob = std::stof(argv[4]);
    int num_steps = std::stoi(argv[5]);
    int num_walks = std::stoi(argv[6]);
    int num_rec = std::stoi(argv[7]);
    int seed = std::stoi(argv[8]);
    
    //Only one randomizer object should be used per MPI rank, and all should have same seed
    Randomizer random_generator(seed, num_nodes, restart_prob);
    int rank, size;

    //Starting MPI pipeline
    MPI_Init(NULL, NULL);
    
    // Extracting Rank and Processor Count
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // create file
    if (rank == 0) {
        std::ofstream fopen("output.dat", std::ios::out | std::ios::binary);
        fopen.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (size <= 2) {
        // initialise user graph
        std::vector<std::vector<int>> G(num_nodes);
        // read edge data from file
        std::ifstream fin(graph_file, std::ios::in | std::ios::binary);
        uchar num_a[4], num_b[4];
        for (int i = 0; i < num_edges; i ++) {
            int a, b;
            if (!misc::is_little_endian()) {
                fin.read((char*) &a, sizeof(int));
                fin.read((char*) &b, sizeof(int));
            } else {
                misc::read_int(&fin, &a);
                misc::read_int(&fin, &b);
            }
            assert(a >= 0 && b >= 0 && a < num_nodes && b < num_nodes);
            G[a].push_back(b);
        }
        // close input file
        fin.close();
        std::vector<int> prefix_sum(num_nodes);
        // compute outdegree prefix sum
        for (int i = 0; i < num_nodes; i ++) {
            if (i == 0) {
                prefix_sum[i] = G[i].size();
            } else {
                prefix_sum[i] = prefix_sum[i - 1] + G[i].size();
            }
        }
        int total_count = prefix_sum[num_nodes - 1];
        int start_count = rank * (total_count / size), end_count = (rank == size - 1) ? (total_count + 1) : (start_count + (total_count / size));
        // divide users among processes, valid_uid = [start_uid, end_uid)
        int start_uid = (int) (std::lower_bound(prefix_sum.begin(), prefix_sum.end(), start_count) - prefix_sum.begin());
        int end_uid = (int) (std::lower_bound(prefix_sum.begin(), prefix_sum.end(), end_count) - prefix_sum.begin());

        // open output file (in append mode)
        std::ofstream fout("output.dat", std::ios::ate | std::ios::binary | std::ios::in);
        int row_size = 4 + 8 * num_rec;
        fout.seekp(start_uid * row_size, std::ios_base::beg);
        // initialise null string
        char nullstr[5] = "NULL";
        // determine influence score
        std::vector<std::pair<int, int>> score(num_nodes);
        // neighbour vector
        std::vector<bool> neighbour(num_nodes);
        for (int i = start_uid; i < end_uid; i ++) {
            // initialise scores
            for (int j = 0; j < num_nodes; j ++) {
                score[j].first = 0;
                score[j].second = j;
                neighbour[j] = (i == j);
            }
            // get number of children
            int num_child = G[i].size();
            // start num_walks rwr from each node in L
            for (int j = 0; j < num_child; j ++) {
                neighbour[G[i][j]] = 1;
                int temp_walks = num_walks;
                while (temp_walks --) {
                    int curr_node = G[i][j];
                    int temp_steps = num_steps;
                    while (temp_steps --) {
                        int out_degree = G[curr_node].size();
                        if (out_degree > 0) {
                            int next_step = random_generator.get_random_value(i);
                            if (next_step < 0) {
                                curr_node = G[i][j];
                            } else {
                                curr_node = G[curr_node][next_step % out_degree];
                            }
                        } else {
                            curr_node = G[i][j];
                        }
                        score[curr_node].first ++;
                    }
                }
            }
            for (int j = 0; j < num_rec; j ++) {
                if (neighbour[j]) {
                    score[j].first = 0;
                }
            }
            std::make_heap(score.begin(), score.begin() + num_rec, misc::comparePairMin());
            for (int j = num_rec; j < num_nodes; j ++) {
                if (neighbour[j]) {
                    continue;
                }
                if (score[j].first > score.front().first) {
                    std::pop_heap(score.begin(), score.begin() + num_rec, misc::comparePairMin());
                    score[num_rec - 1].first = score[j].first;
                    score[num_rec - 1].second = score[j].second;
                    std::push_heap(score.begin(), score.begin() + num_rec, misc::comparePairMin());
                }
            }
            std::vector<std::pair<int, int>> temp;
            for (int j = 0; j < num_rec; j ++) {
                std::pop_heap(score.begin(), score.begin() + num_rec - j, misc::comparePairMin());
                if (score[num_rec - j - 1].first == 0) {
                    continue;
                } else {
                    temp.push_back(std::pair<int, int>(score[num_rec - j - 1].first, score[num_rec - j - 1].second));
                }
            }
            std::reverse(temp.begin(), temp.end());
            // write out-degree to file
            if (!misc::is_little_endian()) {
                fout.write((char*) &num_child, sizeof(int));
            } else {
                misc::write_int(&fout, &num_child);
            }
            // write recommendations to file
            int actual_rec = temp.size();
            for (int j = 0; j < actual_rec; j ++) {
                if (!misc::is_little_endian()) {
                    fout.write((char*) &temp[j].second, sizeof(int));
                    fout.write((char*) &temp[j].first, sizeof(int));
                } else {
                    misc::write_int(&fout, &temp[j].second);
                    misc::write_int(&fout, &temp[j].first);
                }
            }
            while (actual_rec < num_rec) {
                fout.write((char*) nullstr, sizeof(nullstr) - 1);
                fout.write((char*) nullstr, sizeof(nullstr) - 1);
                actual_rec ++;
            }
        }
    } else {
        // initialise user graph
        std::vector<std::vector<int>> G(num_nodes);
        // read edge data from file
        std::ifstream fin(graph_file, std::ios::in | std::ios::binary);
        uchar num_a[4], num_b[4];
        for (int i = 0; i < num_edges; i ++) {
            int a, b;
            if (!misc::is_little_endian()) {
                fin.read((char*) &a, sizeof(int));
                fin.read((char*) &b, sizeof(int));
            } else {
                misc::read_int(&fin, &a);
                misc::read_int(&fin, &b);
            }
            G[a].push_back(b);
        }
        // close input file
        fin.close();
        if (rank == 0) {
            int chunk_size = misc::min(num_nodes / (size - 1), 1500);
            int start_uid = 0; 
            int buff[2];
            for (int i = 1; i < size; i ++) {
                buff[0] = start_uid;
                buff[1] = start_uid + chunk_size;
                MPI_Send((void*) buff, 2, MPI_INT, i, 1, MPI_COMM_WORLD);
                start_uid += chunk_size;
            }
            // status object
            MPI_Status status;
            int message;
            while (start_uid < num_nodes) {
                MPI_Recv((void*) &message, 1, MPI_INT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status);
                buff[0] = start_uid;
                buff[1] = start_uid + chunk_size;
                MPI_Send((void*) buff, 2, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
                start_uid += chunk_size;
            }
            buff[0] = buff[1] = -1;
            int count = 0;
            while (count < size - 1) {
                MPI_Recv((void*) &message, 1, MPI_INT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status);
                MPI_Send((void*) buff, 2, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
                count ++;
            }
        } else {
            // open output file
            std::ofstream fout("output.dat", std::ios::ate | std::ios::binary | std::ios::in);
            int row_size = 4 + 8 * num_rec;
            char nullstr[5] = "NULL";
            // influence score vector
            std::vector<std::pair<int, int>> score(num_nodes);
            // neighbour vector
            std::vector<bool> neighbour(num_nodes);
            // send buffer
            int message = 1;
            // start-end buffer
            int buff[2];
            MPI_Status status;
            // receive initial group of users
            MPI_Recv((void*) buff, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
            int start_uid = buff[0], end_uid = buff[1];
            while (true) {
                fout.seekp(start_uid * row_size, std::ios_base::beg);
                for (int i = start_uid; i < misc::min(end_uid, num_nodes); i ++) {
                    // initialise scores
                    for (int j = 0; j < num_nodes; j ++) {
                        score[j].first = 0;
                        score[j].second = j;
                        neighbour[j] = (i == j);
                    }
                    // get number of children
                    int num_child = G[i].size();
                    // start num_walks rwr from each node in L
                    for (int j = 0; j < num_child; j ++) {
                        neighbour[G[i][j]] = 1;
                        int temp_walks = num_walks;
                        while (temp_walks --) {
                            int curr_node = G[i][j];
                            int temp_steps = num_steps;
                            while (temp_steps --) {
                                int out_degree = G[curr_node].size();
                                if (out_degree > 0) {
                                    int next_step = random_generator.get_random_value(i);
                                    if (next_step < 0) {
                                        curr_node = G[i][j];
                                    } else {
                                        curr_node = G[curr_node][next_step % out_degree];
                                    }
                                } else {
                                    curr_node = G[i][j];
                                }
                                score[curr_node].first ++;
                            }
                        }
                    }
                    for (int j = 0; j < num_rec; j ++) {
                        if (neighbour[j]) {
                            score[j].first = 0;
                        }
                    }
                    std::make_heap(score.begin(), score.begin() + num_rec, misc::comparePairMin());
                    for (int j = num_rec; j < num_nodes; j ++) {
                        if (neighbour[j]) {
                            continue;
                        }
                        if (score[j].first > score.front().first) {
                            std::pop_heap(score.begin(), score.begin() + num_rec, misc::comparePairMin());
                            score[num_rec - 1].first = score[j].first;
                            score[num_rec - 1].second = score[j].second;
                            std::push_heap(score.begin(), score.begin() + num_rec, misc::comparePairMin());
                        }
                    }
                    std::vector<std::pair<int, int>> temp;
                    for (int j = 0; j < num_rec; j ++) {
                        std::pop_heap(score.begin(), score.begin() + num_rec - j, misc::comparePairMin());
                        if (score[num_rec - j - 1].first == 0) {
                            continue;
                        } else {
                            temp.push_back(std::pair<int, int>(score[num_rec - j - 1].first, score[num_rec - j - 1].second));
                        }
                    }
                    std::reverse(temp.begin(), temp.end());
                    // write out-degree to file
                    if (!misc::is_little_endian()) {
                        fout.write((char*) &num_child, sizeof(int));
                    } else {
                        misc::write_int(&fout, &num_child);
                    }
                    // write recommendations to file
                    int actual_rec = temp.size();
                    for (int j = 0; j < actual_rec; j ++) {
                        if (!misc::is_little_endian()) {
                            fout.write((char*) &temp[j].second, sizeof(int));
                            fout.write((char*) &temp[j].first, sizeof(int));
                        } else {
                            misc::write_int(&fout, &temp[j].second);
                            misc::write_int(&fout, &temp[j].first);
                        }
                    }
                    while (actual_rec < num_rec) {
                        fout.write((char*) nullstr, sizeof(nullstr) - 1);
                        fout.write((char*) nullstr, sizeof(nullstr) - 1);
                        actual_rec ++;
                    }                    
                }
                MPI_Send((void*) &message, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
                MPI_Recv((void*) buff, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
                start_uid = buff[0];
                end_uid = buff[1];
                if (start_uid < 0) {
                    break;
                }
            }
        }
    }
    MPI_Finalize();
}