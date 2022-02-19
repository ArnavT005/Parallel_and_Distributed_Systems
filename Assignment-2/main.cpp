#include <string>
#include <mpi.h>
#include <assert.h>
#include <fstream>
#include "randomizer.hpp"

namespace ds {
    
    template <typename T>
    class Array {
        private:
            T *data;
            int size, back;
        public:
            Array();
            Array(int);
            int get_size();
            T operator[](int);
            bool push_back(T);
    };

    template <typename T>
    Array<T>::Array() {
        back = 0;
        size = 1;
        data = new T[1];
    }

    template <typename T>
    Array<T>::Array(int _size) {
        back = 0;
        size = _size;
        data = new T[size];
    }

    template <typename T>
    int Array<T>::get_size() {
        return size;
    }

    template <typename T>
    T Array<T>::operator[](int index) {
        if (0 <= index && index < size) {
            return data[index];
        } else {
            return data[back];
        }
    }

    template <typename T>
    bool Array<T>::push_back(T item) {
        if (back == size) {
            size *= 2;
            T *newdata = new T[size];
            if (newdata == nullptr) {
                return false;
            } else {
                for (int i = 0; i < back; i ++)
                    newdata[i] = data[i];
                newdata[back ++] = item;
                delete[] data;
                data = newdata;
                return true;
            }
        }
        else {
            data[back ++] = item;
            return true;
        }
    }

    class Graph {
        private:
            int num_nodes;
            int num_edges;
            Array<Array<int>> adj_list;
        public:
            Graph(int, int);
            int get_nodes();
            int get_edges();
            Array<int> operator[](int);
    };

    Graph::Graph(int n, int m) {
        num_nodes = n;
        num_edges = m;
        adj_list = Array<Array<int>>(num_nodes);
    }

    int Graph::get_nodes() {
        return num_nodes;
    }

    int Graph::get_edges() {
        return num_edges;
    }

    Array<int> Graph::operator[](int index) {
        if (index < num_nodes) {
            return adj_list[index];
        } else {
            return adj_list[num_nodes - 1];
        }
    }

}

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

    //print_random(rank, num_nodes, random_generator);

    // initialise user graph
    ds::Graph G(num_nodes, num_edges);
    // read graph data from file
    std::ifstream fin(graph_file, std::ios::in | std::ios::binary);
    for (int i = 0; i < num_edges; i ++) {
        int a, b;
        if (!misc::is_little_endian()) {
            fin.read((char*) &a, sizeof(int));
            fin.read((char*) &b, sizeof(int));
            std::cout << a << " " << b << std::endl;
        } else {
            char number[4];
            // read 'a' byte-wise
            for (int i = 0; i < 4; i ++) {
                fin.read(&number[i], sizeof(char));
            }
            a = (((int) number[0]) << 24) | (((int) number[1]) << 16) | (((int) number[2]) << 8) | ((int) number[3]);
            // read 'b' byte-wise
            for (int i = 0; i < 4; i ++) {
                fin.read(&number[i], sizeof(char));
            }
            b = (((int) number[0]) << 24) | (((int) number[1]) << 16) | (((int) number[2]) << 8) | ((int) number[3]);
        }
        bool flag = G[a].push_back(b);
        assert(flag);
    }

    // divide users among processes, valid_uid = [start_uid, end_uid)
    int start_uid = rank * (num_nodes / size), end_uid = (rank == size - 1) ? num_nodes : (start_uid + (num_nodes / size));


    MPI_Finalize();
}