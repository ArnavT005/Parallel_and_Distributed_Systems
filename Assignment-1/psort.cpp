#include "psort.h"
#include <omp.h>

// Generic Dynamic Array
template<typename T>
struct dynamic
{
    T *data;
    uint64_t back, size;

    dynamic() {
        data = new T[1];
        back = 0;
        size = 1;
    }

    void push_back(T item) {
        if (back == size) {
            size *= 2;
            T *newdata = new T[size];
            for (uint64_t i = 0; i < back; i ++)
                newdata[i] = data[i];
            newdata[back ++] = item;
            delete[] data;
            data = newdata;
        }
        else {
            data[back ++] = item;
        }
    }
};

// Partition Array
int64_t partition(uint32_t *data, uint32_t n, uint32_t pivot) {
    int64_t i = 0, j = n - 1;
    uint32_t temp = 0;
    while (i <= j) {
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
void SequentialSort(uint32_t *data, uint32_t n) {
    if (n <= 1)
        return;
    uint32_t pivot = data[0];
    uint32_t p = 1 + partition(data + 1, n - 1, pivot);
    uint32_t temp = data[p];
    data[p] = data[0];
    data[0] = temp;
    SequentialSort(data, p);
    SequentialSort(data + p + 1, n - p - 1);
}

void ParallelSort(uint32_t *data, uint32_t n, int p)
{
    // Entry point to your sorting implementation.
    // Sorted array should be present at location pointed to by data.
    // threshold for shifting to sequential sort
    if (n <= p * p) {
        SequentialSort(data, n);
        return;
    }
    uint32_t *splitters = new uint32_t[p * p];
    uint32_t rem = n % p, size = n / p;
    uint64_t index = 0, count = 0, bucket_num = 0;
    for (int i = 0; i < p * p; i ++) {
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
    dynamic<uint32_t> *partitions = new dynamic<uint32_t>[p];
    uint64_t threshold = 2 * (n / p);
    for (int i = 0; i < p; i ++) {
        #pragma omp task firstprivate(i) shared(partitions)
        {
            uint32_t lower = (i == 0) ? 0 : splitters[i * p];
            uint32_t upper = (i == p - 1) ? 0xffffffff : splitters[(i + 1) * p];
            for (uint64_t j = 0; j < n; j ++) {
                if (i == 0) {
                    if (data[j] >= lower && data[j] <= upper) {
                        partitions[i].push_back(data[j]);
                    }
                }
                else {
                    if (data[j] > lower && data[j] <= upper) {
                        partitions[i].push_back(data[j]);
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
    for (int i = 0; i < p; i ++) {
        for (uint64_t j = 0; j < partitions[i].back; j ++) {
            data[index ++] = partitions[i].data[j];
        }
    }
    delete[] splitters;
}