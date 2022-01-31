#include <iostream>
#include <fstream>
#include <stdint.h>
#include <chrono>
#include <omp.h>
#include "psort.h"
#include <algorithm>


void check_sorted(uint32_t *data, int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        if (data[i] > data[i + 1])
        {
            std::cout << "Data is not sorted.\n";
            return;
        }
    }

    std::cout << "Data is sorted.\n";
    return;
}


void SortData(uint32_t *data, uint32_t n, int p, int n_threads)
{
    #pragma omp parallel num_threads(n_threads)
    {
        #pragma omp single
        ParallelSort(data, n, p);
    }
}


int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Insufficient Arguments\n"; // <datafile> <n_threads>
        return 0;
    }

    // std::fstream fs(argv[1], std::ios::in | std::ios::binary);
    std::fstream fs(argv[1], std::fstream::in);

    // Reading data
    uint32_t n, d = 0;
    int p, n_threads = atoi(argv[2]);
    fs >> n >> p;
    // fs.read((char *)&n, sizeof(n));
    // fs.read((char *)&p, sizeof(p));
    uint32_t *data = new uint32_t[n];
    uint32_t *odata = new uint32_t[n];

    std::cout << "n_threads = " << n_threads << std::endl;
    std::cout << "N = " << n << " p = " << p << std::endl;

    while (fs >> data[d++]) {
        odata[d - 1] = data[d - 1];
        if(d == n)
            break;
    }
    // for (d = 0; d < n; d++) {
    //     fs.read((char *)&data[d], sizeof(uint32_t));
    //     odata[d] = data[d];
    // }
    fs.close();
    std::sort(odata, odata + n);
    // Sorting
    auto begin = std::chrono::high_resolution_clock::now();
    SortData(data, n, p, n_threads);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    double duration = (1e-6 * (std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)).count());

    check_sorted(data, n);
    bool flag = false;
    for(uint64_t i = 0; i < n; i ++) {
        if(data[i] != odata[i])
            flag = true;
    }
    std::cout << "Time taken for sorting " << n << " elements with "
        << p << " buckets = " << duration << "ms" << std::endl;
    if(!flag)
        std::cout << "CORRECT!" << std::endl;
    // Clean-up
    delete data;

    return 0;
}