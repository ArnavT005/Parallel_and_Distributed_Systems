#include "classify.h"
#include <omp.h>
#include <vector>
#include <utility>
#include <chrono>

Range::Range(int a=1, int b=0) { // Constructor. Defaults to *bad* range
   lo = a;
   hi = b;
}

bool Range::within(int val) const { // Return if val is within this range
      return(lo <= val && val <= hi);
}

bool Range::strictlyin(int val) const { // Return if val is strictly inside this range
      return(lo < val && val < hi);
}

void Ranges::set(int i, int lo, int hi) { // set the extreme values of a specific interval
   if(i < _num) {
      _ranges[i].lo = lo;
      _ranges[i].hi = hi;
   }
}

Ranges::Ranges() { // Initialize with a single unreal interval
   _num = 1;
   _ranges = new Range(1, 0); // Started with this. Its not a real interval as nothing lies inside it.
}

Ranges& Ranges::operator+=(const Range range){ // Add one more interval to this list
   if(newrange(range)) { // If it already exists, do not add
      Range *oranges = _ranges;
      _ranges = new Range[_num+1];
      assert(NULL != _ranges);
      for(int r=0; r<_num; r++) { 
         set(r, oranges[r].lo, oranges[r].hi); // copy old intervals
      }
      set(_num++, range.lo, range.hi); // Add the new interval at the end
   }
   return *this;
}

int Ranges::range_binary(int val, bool strict = false) const {
   int low = 0, high = _num - 1, mid = 0;
   if(strict) {
      while(low <= high) {
         mid = (low + high) / 2;
         if(_ranges[mid].strictlyin(val))
            return mid;
         else if(val >= _ranges[mid].hi) {
            low = mid + 1;
         }
         else {
            high = mid  - 1;
         }
      }
   } else {
      while(low <= high) {
         mid = (low + high) / 2;
         if(_ranges[mid].within(val))
            return mid;
         else if(val > _ranges[mid].hi) {
            low = mid + 1;
         }
         else {
            high = mid - 1;
         }
      }
   }
   return BADRANGE; // Did not find any range
}

int Ranges::range(int val, bool strict = false) const { // Tell the range in which val lies (strict => boundary match not ok)
   if(strict) {
      for(int r=0; r<_num; r++) // Look through all intervals
         if(_ranges[r].strictlyin(val))
            return r;
   } else {
      for(int r=0; r<_num; r++) // Look through all intervals
         if(_ranges[r].within(val))
            return r;
   }
   return BADRANGE; // Did not find any range
}

void Ranges::inspect() {
   for(int r=0; r<_num; r++) { 
      std::cout << r << "," << &_ranges[r] << ": " << _ranges[r].lo << ", " << _ranges[r].hi << "\n"; 
   }

}

int Ranges::num() const { return _num; }


bool Ranges::newrange(const Range r) { // Is the range r already in my list, or is it a new one?
   return (range(r.lo, true) == BADRANGE && range(r.hi, true) == BADRANGE); // Overlaps are not allowed.
}

Item::Item() {
   key = value = -1;
}

Item::Item(int a, int b) {
   key = a;
   value = b;
}

Data classify(Data &D, const Ranges &R, unsigned int numt)
{  
   // Classify each item in D into intervals (given by R). Finally, produce in D2 data sorted by interval
   assert(numt < MAXTHREADS);
   
   auto begin = std::chrono::high_resolution_clock::now();
   std::vector<std::vector<unsigned int>> counts(numt, std::vector<unsigned int>(R.num(), 0));
   std::vector<std::vector<int>> range(numt, std::vector<int>((int)(D.ndata / numt) + 1, 0));
   #pragma omp parallel num_threads(numt)
   {
      int tid = omp_get_thread_num(); // I am thread number tid
      for(int i=tid; i<D.ndata; i+=numt) { // Threads together share-loop through all of Data
         int v = range[tid][i / numt] = R.range(D.data[i].key);// For each data, find the interval of data's key,
							                                          // and store the interval id in value. D is changed.
         counts[tid][v] ++;
      }
   }
   auto end = std::chrono::high_resolution_clock::now();
   auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
   std::cout << "First pragma " << (1e-6 * elapsed.count()) <<  " ms" << std::endl;
   // int k = 4;
   // #pragma omp parallel num_threads(numt)
   // {
   //    int tid = omp_get_thread_num(); // I am thread number tid
   //    int index = k * tid;
   //    while(index < D.ndata) {
   //       for(int j = index; j < (index + k) && j < D.ndata; j ++) {
   //          int v = range[tid][(j / (k * numt)) * k + j % k] = R.range(D.data[j].key);
   //          counts[tid][v] ++;
   //       }
   //       index += k * numt;
   //    }
   //    // tid == (index / k) % numt
   //    // f(index) == (index / (k * numt)) * k + index % k
   // }

   auto begin1 = std::chrono::high_resolution_clock::now();
   // Accumulate all sub-counts (in each interval's counter) into rangecount
   unsigned int *rangecount = new unsigned int[R.num()]();
   std::vector<int> partition_size(numt, 0);
   int p = -1;
   for(int t = 0; t < numt; t ++) {
      for(int r = 1; r < R.num(); r ++) {
         rangecount[r] += counts[t][r];   
         if(r / (R.num() / numt) >= numt)
            p = numt - 1;
         else
            p = r / (R.num() / numt);
         partition_size[p] += counts[t][r];
      }
   }
   for(int r = 1; r < R.num(); r ++) {
      rangecount[r] += rangecount[r - 1];
   }
   auto end1 = std::chrono::high_resolution_clock::now();
   auto elapsed1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - begin1);
   std::cout << "Rangecount sum " << (1e-6 * elapsed1.count()) <<  " ms" << std::endl;
   // Now rangecount[i] has the number of elements in intervals before the ith interval.

   
   auto begin2 = std::chrono::high_resolution_clock::now();
   std::vector<std::vector<std::pair<int, int>>> partitions(numt, std::vector<std::pair<int, int>>(0));
   for(int i = 0; i < numt; i ++) {
      partitions[i].resize(partition_size[i]);
   }
   std::vector<int> threadIndex(numt, 0);
   for(int i = 0; i < D.ndata; i ++) {
      //int r = 1, p = -1;
      int r = range[i % numt][i / numt], p = -1;
      // int r = range[(i / k) % numt][(i / (k * numt)) * k + i % k], p = -1;
      if(r / (R.num() / numt) >= numt)
         p = numt - 1;
      else
         p = r / (R.num() / numt);
      // if(partition_size[p] == partitions[p].size())
      //    partitions[p].push_back({D.data[i].key, r});
      //    // partitions[p].push_back(Item(D.data[i].key, r));
      // else
      //    partitions[p][partition_size[p]] = {D.data[i].key, r};
      //    // partitions[p][partition_size[p]] = Item(D.data[i].key, r);
      //partition_size[p] ++;
      partitions[p][threadIndex[p] ++] = {D.data[i].key, r};

   }
   auto end2 = std::chrono::high_resolution_clock::now();
   auto elapsed2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - begin2);
   std::cout << "Partition determination " << (1e-6 * elapsed2.count()) <<  " ms" << std::endl;

   Data D2 = Data(D.ndata); // Make a copy
   auto begin5 = std::chrono::high_resolution_clock::now();
   std::vector<std::vector<std::pair<int, int>>> items(numt, std::vector<std::pair<int, int>>(0));
   for(int i = 0; i < numt; i ++) {
      items[i].resize(partition_size[i]);
      if(i != 0) 
         partition_size[i] += partition_size[i - 1];
   }
   auto end5 = std::chrono::high_resolution_clock::now();
   auto elapsed5 = std::chrono::duration_cast<std::chrono::nanoseconds>(end5 - begin5);
   std::cout << "Dummy " << (1e-6 * elapsed5.count()) <<  " ms" << std::endl;
   
   auto begin3 = std::chrono::high_resolution_clock::now();
   #pragma omp parallel num_threads(numt)
   {
      int tid = omp_get_thread_num();
      int lower = tid * (R.num() / numt), upper = (tid + 1 == numt) ? R.num() : (tid + 1) * (R.num() / numt);
      std::vector<int> rangeIndex(upper - lower, 0);
      int temp = (lower == 0) ? 0 : rangecount[lower - 1];
      for(int i = 0; i < partitions[tid].size(); i ++) {
         int r = partitions[tid][i].second;
         // int r = partitions[tid][i].value;
         if(r == -1) 
            break;
         items[tid][rangecount[r - 1] - temp + rangeIndex[r - lower] ++] = {partitions[tid][i].first, r};
         // items[tid][rangecount[r - 1] - temp + rangeIndex[r - lower] ++] = Item(partitions[tid][i].key, r);
      } 
   }
   auto end3 = std::chrono::high_resolution_clock::now();
   auto elapsed3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - begin3);
   std::cout << "Second pragma " << (1e-6 * elapsed3.count()) <<  " ms" << std::endl;

   auto begin4 = std::chrono::high_resolution_clock::now();
   #pragma omp parallel num_threads(numt)
   {
      int tid = omp_get_thread_num();
      for(int i = 0; i < items[tid].size(); i ++) {
         if(tid == 0) {
            // D2.data[i] = items[tid][i];
            D2.data[i].key = items[tid][i].first;
            D2.data[i].value = items[tid][i].second;
         }
         else {
            // D2.data[partition_size[tid - 1] + i] = items[tid][i];
            D2.data[partition_size[tid - 1] + i].key = items[tid][i].first;
            D2.data[partition_size[tid - 1] + i].value = items[tid][i].second;
         }
      }

   }
   auto end4 = std::chrono::high_resolution_clock::now();
   auto elapsed4 = std::chrono::duration_cast<std::chrono::nanoseconds>(end4 - begin4);
   std::cout << "Third pragma " << (1e-6 * elapsed4.count()) <<  " ms" << std::endl;

   // int globalIndex = 0, tid = 0, tIndex = 0;
   // while(globalIndex < D2.ndata) {
   //    if(tIndex < partitions[tid].size()) {
   //       D2.data[globalIndex].key = items[tid][tIndex].first;
   //       D2.data[globalIndex].value = items[tid][tIndex].second;
   //       tIndex ++;
   //       globalIndex ++;
   //    }
   //    else {
   //       tid ++;
   //       tIndex = 0;
   //    }
   // }
   return D2;
}
