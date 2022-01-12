#include "classify_new.h"
#include <omp.h>
#include <vector>
#include <mutex>

int min(const int &a, const int &b) {
   if(a <= b) {
      return a;
   }
   else {
      return b;
   }
}

Data classify(Data &D, const Ranges &R, unsigned int numt)
{  
   // Classify each item in D into intervals (given by R). Finally, produce in D2 data sorted by interval
   assert(numt < MAXTHREADS);
   Counter counts[R.num()]; // I need on counter per interval. Each counter can keep pre-thread subcount.
   std::vector<int> offset(numt);
   for(int i = 0; i < numt; i ++) {
      offset[i] = 8 * i;
   }
   #pragma omp parallel num_threads(numt)
   {
      int tid = omp_get_thread_num(); // I am thread number tid
      // int index = 0;
      // while(index < D.ndata) {
      //    for(int i = index + offset[tid]; i < min(index + offset[tid] + 8, D.ndata); i ++) {
      //       int v = D.data[i].value = R.range_binary(D.data[i].key);
      //       counts[v].increase(tid);
      //    }
      //    index += numt * 8;
      // }

      for(int i=tid; i<D.ndata; i+=numt) { // Threads together share-loop through all of Data
         int v = D.data[i].value = R.range_binary(D.data[i].key);// For each data, find the interval of data's key,
							  // and store the interval id in value. D is changed.
         counts[v].increase(tid); // Found one key in interval v
      }
   }

   // Optimisation 3: (Minor) Compute prefix sum and accumulate sub-parts simultaneously
   // Accumulate all sub-counts (in each interval;'s counter) into rangecount
   // Also compute prefix sum
   // unsigned int *rangecount = new unsigned int[R.num()];
   // rangecount[0] = 0;
   // for(int r=1; r<R.num(); r++) { // For all intervals
   //    rangecount[r] = 0;
   //    for(int t=0; t<numt; t++) // For all threads
   //       rangecount[r] += counts[r].get(t);
   //    rangecount[r] += rangecount[r - 1];
   // }

   // Now rangecount[i] has the number of elements in intervals before the ith interval.

   // std::mutex darrayMutex;
   // unsigned int *rangeIndex = new unsigned int[R.num()]();
   Data D2 = Data(D.ndata); // Make a copy

   // std::vector<std::vector<Data::Item>> subArray(numt, std::vector<Data::Item>());
   // for(int i = 0; i < D.ndata; i ++) {
   //    Data::Item item(D.data[i].key, D.data[i].value);
   //    for(int j = 0; j < numt; j ++) {
   //       if(D.data[i].value >= (j / numt) * R.num() && D.data[i].value < ((j + 1) / numt) * R.num()) {
   //          subArray[j].push_back(item);
   //          break;
   //       }
   //    }
   // }
   // #pragma omp parallel num_threads(numt)
   // {
   //    int tid = omp_get_thread_num(), r = 0;
   //    for(int i = 0; i < subArray[tid].size(); i ++) {
   //       r = subArray[tid][i].value;
   //       D2.data[rangecount[r - 1] + rangeIndex[r] ++] = subArray[tid][i];
   //    }  
   // }
   // #pragma omp parallel num_threads(numt)
   // {
   //    int tid = omp_get_thread_num();
   //    // int index = 0;
   //    // while(index < R.num()) {
   //    //    for(int r=index + offset[tid]; r<min(index + offset[tid] + 8, R.num()); r++) { // Thread together share-loop through the intervals 
   //    //       int rcount = 0;
   //    //       for(int d=0; d<D.ndata; d++) // For each interval, thread loops through all of data and  
   //    //          if(D.data[d].value == r) // If the data item is in this interval 
   //    //             D2.data[rangecount[r-1]+rcount++] = D.data[d]; // Copy it to the appropriate place in D2.
   //    //    }
   //    //    index += 8 * numt;
   //    // }
   //    for(int r=tid; r<R.num(); r+=numt) { // Thread together share-loop through the intervals 
   //       alignas(64) int rcount = 0;
   //       for(int d=0; d<D.ndata; d++) // For each interval, thread loops through all of data and  
   //           if(D.data[d].value == r) // If the data item is in this interval 
   //               D2.data[rangecount[r-1]+rcount++] = D.data[d]; // Copy it to the appropriate place in D2.
   //    }
   // }
   // #pragma omp parallel num_threads(numt)
   // {
   //    int tid = omp_get_thread_num();
   //    // Optimisation 4: Improve loop structure. Avoid redundancy
   //    int index = 0;
   //    while(index < D.ndata) {
   //       for(int i = index + offset[tid]; i < min(index + offset[tid] + 8, D.ndata); i ++) {
   //          int r = D.data[i].value;
   //          //std::lock_guard<std::mutex> lock(darrayMutex);
   //          D2.data[rangecount[r - 1] + rangeIndex[r]++] = D.data[i];
   //       }
   //       index += numt * 8;
   //    }
   //    // for(int i = tid; i < D.ndata; i += numt) {
   //    //    int r = D.data[i].value;
   //    //    std::lock_guard<std::mutex> lock(darrayMutex);
   //    //    D2.data[rangecount[r - 1] + rangeIndex[r]++] = D.data[i];
   //    // }
   // }
   return D2;
}
