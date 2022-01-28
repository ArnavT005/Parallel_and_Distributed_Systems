#include "psort.h"
#include <omp.h>
#include <vector>
#include <assert.h>
#include <iostream>

template<typename T>
struct dynamic
{
    T *data;
    uint32_t back, size;

    dynamic() {
        data = new T[1];
        back = 0;
        size = 1;
    }

    void push_back(T item) {
        if (back == size) {
            size *= 2;
            T *newdata = new T[size];
            for (int i = 0; i < back; i ++)
                *(newdata + i) = *(data + i);
            assert(back < size);
            *(newdata + back) = item;
            delete[] data;
            back ++;
            data = newdata;
        }
        else {
            assert(back < size);
            *(data + back) = item;
            back ++;
        }
    }
};

// Partition Array
int32_t partition(uint32_t *data, uint32_t n, uint32_t pivot) {
    assert(n > 0);
    int32_t i = 0, j = n - 1;
    uint32_t temp = 0;
    while (i <= j) {
        assert(i < n && i >= 0);
        assert(j < n && j >= 0);
        if (data[i] <= pivot)
            i ++;
        else {
            temp = data[i];
            data[i] = data[j];
            data[j] = temp;
            j --;
        }
    }
    return j;
}

// Quick Sort
// Optimisation: Randomise
void SequentialSort(uint32_t *data, uint32_t n) {
    if (n <= 1)
        return;
    uint32_t pivot = *(data);
    int32_t p = 1 + partition(data + 1, n - 1, pivot);
    assert(p < n && p >= 0);
    uint32_t temp = data[p];
    data[p] = data[0];
    data[0] = temp;
    // std::cout << "Hello" << std::endl;
    SequentialSort(data, p);
    SequentialSort(data + p + 1, n - p - 1);
}

void ParallelSort(uint32_t *data, uint32_t n, int p)
{
    // Entry point to your sorting implementation.
    // Sorted array should be present at location pointed to by data.
    // threshold for shifting to sequential sort
    // check if pseudo-splitters can be found
    // if not, sort sequentially
    if (n <= p * p) {
        SequentialSort(data, n);
    }
    uint32_t *splitters = new uint32_t[p * p];
    uint32_t rem = n % p, size = n / p;
    uint32_t index = 0, count = 0, bucket_num = 0;
    for (int i = 0; i < p * p; i ++) {
        assert(index < n);
        splitters[i] = data[index];
        count ++;
        index ++;
        if (count == p) {
            bucket_num ++;
            if (bucket_num < rem)
                index = (size + 1) * bucket_num;
            else
                index = (size + 1) * rem + (bucket_num - rem) * size;
            count = 0;
        }
    }
    
    SequentialSort(splitters, p * p);
    // s_j = splitters[(j + 1) * p], 0 <= j <= p - 2
    std::vector<dynamic<uint32_t>> partitions(p - 1);
    uint32_t threshold = 2 * n / p;
    for (int i = 0; i < p - 1; i ++) {
        #pragma omp task firstprivate(i) shared(partitions)
        {
            assert(i * p < p * p);
            assert((i + 1) * p < p * p);
            uint32_t lower = (i == 0) ? 0 : splitters[i * p];
            uint32_t upper = (i == p - 2) ? 0xffffffff : splitters[(i + 1) * p];
            for (int j = 0; j < n; j ++) {
                if (i == 0) {
                    if (data[j] >= lower && data[j] <= upper) {
                        partitions[i].push_back(*(data + j));
                    }
                }
                else {
                    if (data[j] > lower && data[j] <= upper) {
                        partitions[i].push_back(*(data + j));
                    }
                }
            }
            if (partitions[i].back < threshold) {
                SequentialSort(partitions[i].data, partitions[i].back);
            }
            else 
                ParallelSort(partitions[i].data, partitions[i].back, p);
        }
    }
    #pragma omp taskwait
    index = 0;
    for (int i = 0; i < p - 1; i ++) {
        for (int j = 0; j < partitions[i].back; j ++) {
            assert(index < n);
            *(data + index) = *(partitions[i].data + j);
            index ++;
        }
    }
    delete[] splitters;
    //check_sorted1(data, n);
}