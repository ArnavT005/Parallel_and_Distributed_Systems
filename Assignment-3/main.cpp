#include <bits/stdc++.h>
#include <mpi.h>

template <typename T>
using V = std::vector<T>;

template <typename T, typename K>
using P = std::pair<T, K>;

namespace metric {
    double norm(V<double> &a) {
        double norm = 0;
        for (int i = 0; i < a.size(); i ++) {
            norm += a[i] * a[i];
        }
        return sqrt(norm);
    }

    double cosine_dist(V<double> &a, V<double> &b) {
        double dist = 0;
        for (int i = 0; i < a.size(); i ++) {
            dist += (a[i] * b[i]);
        }
        return dist / (norm(a) * norm(b));
    }
}

namespace hnsw {
    struct comparePairMax {
        bool operator() (const P<int, double> &a, const P<int, double> &b) {
            if (a.second  < b.second) {
                return true;
            } else if (a.second == b.second) {
                return b.first < a.first;
            } else {
                return false;
            }
        }
    };
    
    V<P<int, double>> SearchLayer(V<double> &user, V<P<int, double>> &candidates, V<int> &indptr, V<int> &index, V<int> &level_offset, int level, V<bool> &visited, V<V<double>> vect, int K) {
        V<P<int, double>> topk = candidates;
        while (candidates.size() > 0) {
            std::pop_heap(candidates.begin(), candidates.end(), comparePairMax());
            int ep = candidates[candidates.size() - 1].first;
            candidates.pop_back();
            int start = indptr[ep] + level_offset[level], end = indptr[ep] + level_offset[level + 1];
            for (int i = start; i < end; i ++) {
                if (visited[index[i]] || index[i] == -1) {
                    continue;
                }
                visited[index[i]] = true;
                double _dist = metric::cosine_dist(user, vect[index[i]]);
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

    V<P<int, double>> QueryHNSW(V<double> &user, int ep, V<int> &indptr, V<int> &index, V<int> &level_offset, int max_level, V<V<double>> vect, int K) {
        V<P<int, double>> topk;
        topk.push_back(P<int, double>(ep, metric::cosine_dist(user, vect[ep])));
        V<bool> visited(vect.size(), false);
        visited[ep] = true;
        for (int level = max_level; level >= 0; level --) {
            topk = SearchLayer(user, topk, indptr, index, level_offset, level, visited, vect, K);
        }
        return topk;
    }
}

int main(int argc, char **argv) {
    std::string out_dir = argv[1];
    int K = std::stoi(argv[2]);
    std::string in_file = argv[3];
    std::string out_file = argv[4];
    int max_level, ep;
    //read max_level
    std::ifstream fin(out_dir + "/max_level.txt", std::ios::in);
    fin >> max_level;
    fin.close();
    //read ep
    fin.open(out_dir + "/ep.txt", std::ios::in);
    fin >> ep;
    fin.close();
    V<int> level, index, indptr, level_offset;
    //read level
    fin.open(out_dir + "level.txt", std::ios::in);
    std::string line;
    while (std::getline(fin, line)) {
        std::stringstream stream(line);
        int temp;
        while (stream >> temp) {
            level.push_back(temp);
        }
    }
    fin.close();
    //read index
    fin.open(out_dir + "/index.txt", std::ios::in);
    while (std::getline(fin, line)) {
        std::stringstream stream(line);
        int temp;
        while (stream >> temp) {
            index.push_back(temp);
        }
    }
    fin.close();
    //read indptr
    fin.open(out_dir + "/indptr.txt", std::ios::in);
    while (std::getline(fin, line)) {
        std::stringstream stream(line);
        int temp;
        while (stream >> temp) {
            indptr.push_back(temp);
        }
    }
    //read level_offset
    fin.open(out_dir + "/level_offset.txt", std::ios::in);
    while (std::getline(fin, line)) {
        std::stringstream stream(line);
        int temp;
        while (stream >> temp) {
            level_offset.push_back(temp);
        }
    }
    fin.close();
    //read vect
    V<V<double>> vect;
    fin.open(out_dir + "/vect.txt", std::ios::in);
    while (std::getline(fin, line)) {
        vect.push_back(V<double>());
        std::stringstream stream(line);
        double temp;
        while (stream >> temp) {
            vect[vect.size() - 1].push_back(temp);
        }
    }
    fin.close();
    //initialize MPI pipeline
    int size, rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //read users.bin and divide users
    MPI_Finalize();
}