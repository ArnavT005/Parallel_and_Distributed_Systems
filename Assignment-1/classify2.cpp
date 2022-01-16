#include "classify.h"
#include <omp.h>
#include <vector>
#include <utility>
#include <chrono>

Range::Range(int a=1, int b=0) {
   lo = a;
   hi = b;
}

bool Range::within(int val) const {
      return(lo <= val && val <= hi);
}

bool Range::strictlyin(int val) const {
      return(lo < val && val < hi);
}

void Ranges::set(int i, int lo, int hi) {
   if(i < _num) {
      _ranges[i].lo = lo;
      _ranges[i].hi = hi;
   }
}

Ranges::Ranges() {
   _num = 1;
   _ranges = new Range(1, 0);
}

Ranges& Ranges::operator+=(const Range range){
   if(newrange(range)) {
      Range *oranges = _ranges;
      _ranges = new Range[_num+1];
      assert(NULL != _ranges);
      for(int r=0; r<_num; r++) { 
         set(r, oranges[r].lo, oranges[r].hi);
      }
      set(_num++, range.lo, range.hi);
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
   return BADRANGE;
}

int Ranges::range(int val, bool strict = false) const {
   if(strict) {
      for(int r=0; r<_num; r++)
         if(_ranges[r].strictlyin(val))
            return r;
   } else {
      for(int r=0; r<_num; r++)
         if(_ranges[r].within(val))
            return r;
   }
   return BADRANGE;
}

void Ranges::inspect() {
   for(int r=0; r<_num; r++) { 
      std::cout << r << "," << &_ranges[r] << ": " << _ranges[r].lo << ", " << _ranges[r].hi << "\n"; 
   }
}

int Ranges::num() const { return _num; }


bool Ranges::newrange(const Range r) {
   return (range(r.lo, true) == BADRANGE && range(r.hi, true) == BADRANGE);
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
   assert(numt < MAXTHREADS);
   
   std::vector<std::vector<unsigned int>> counts(numt, std::vector<unsigned int>(R.num(), 0));
   // std::vector<std::vector<int>> range(numt, std::vector<int>((int)(D.ndata / numt) + 1, 0));
   // #pragma omp parallel num_threads(numt)
   // {
   //    int tid = omp_get_thread_num();
   //    for(int i=tid; i<D.ndata; i+=numt) { 
   //       // int v = range[tid][i / numt] = R.range(D.data[i].key);
   //       int v = D.data[i].value = R.range(D.data[i].key);
   //       counts[tid][v] ++;
   //    }
   // }

   int k = 12;
   #pragma omp parallel num_threads(numt)
   {
      int tid = omp_get_thread_num(); // I am thread number tid
      int index = k * tid;
      while(index < D.ndata) {
         for(int j = index; j < (index + k) && j < D.ndata; j ++) {
            // int v = range[tid][(j / (k * numt)) * k + j % k] = R.range(D.data[j].key);
            int v = D.data[j].value = R.range(D.data[j].key);
            counts[tid][v] ++;
         }
         index += k * numt;
      }
   }

   unsigned int *rangecount = new unsigned int[R.num()]();
   // std::vector<int> partition_size(numt, 0);
   int p = -1;
   for(int t = 0; t < numt; t ++) {
      for(int r = 1; r < R.num(); r ++) {
         rangecount[r] += counts[t][r];   
         // if(r / (R.num() / numt) >= numt)
         //    p = numt - 1;
         // else
         //    p = r / (R.num() / numt);
         // partition_size[p] += counts[t][r];
      }
   }
   for(int r = 1; r < R.num(); r ++) {
      rangecount[r] += rangecount[r - 1];
   }
   
   std::vector<std::vector<std::pair<int, int>>> partitions(numt, std::vector<std::pair<int, int>>(0));
   // for(int i = 0; i < numt; i ++) {
   //    partitions[i].resize(partition_size[i]);
   // }
   // std::vector<int> threadIndex(numt, 0);
   // for(int i = 0; i < D.ndata; i ++) {
   //    // int r = range[i % numt][i / numt], p = -1;
   //    int r = D.data[i].value, p = -1;
   //    // int r = range[(i / k) % numt][(i / (k * numt)) * k + i % k], p = -1;
   //    if(r / (R.num() / numt) >= numt)
   //       p = numt - 1;
   //    else
   //       p = r / (R.num() / numt);
   //    partitions[p][threadIndex[p] ++] = {D.data[i].key, r};
   // }

   Data D2 = Data(D.ndata); // Make a copy
   unsigned int *rangeIndex = new unsigned int[R.num()]();
   #pragma omp parallel num_threads(numt)
   {
      int tid = omp_get_thread_num();
      for(int d = 0; d < D.ndata; d ++) {
          int r = D.data[d].value;
          if(tid == r % numt) {
              D2.data[rangecount[r - 1] + rangeIndex[r] ++] = D.data[d];
          }
      }
   }
   // std::vector<std::vector<std::pair<int, int>>> items(numt, std::vector<std::pair<int, int>>(0));
   // for(int i = 0; i < numt; i ++) {
   //    items[i].resize(partition_size[i]);
   //    if(i != 0) 
   //       partition_size[i] += partition_size[i - 1];
   // }
   
   // #pragma omp parallel num_threads(numt)
   // {
   //    int tid = omp_get_thread_num();
   //    int lower = tid * (R.num() / numt), upper = (tid + 1 == numt) ? R.num() : (tid + 1) * (R.num() / numt);
   //    std::vector<int> rangeIndex(upper - lower, 0);
   //    int temp = (lower == 0) ? 0 : rangecount[lower - 1];
   //    for(int i = 0; i < partitions[tid].size(); i ++) {
   //       int r = partitions[tid][i].second;
   //       if(r == -1) 
   //          break;
   //       items[tid][rangecount[r - 1] - temp + rangeIndex[r - lower] ++] = {partitions[tid][i].first, r};
   //    } 
   // }

   // #pragma omp parallel num_threads(numt)
   // {
   //    int tid = omp_get_thread_num();
   //    for(int i = 0; i < items[tid].size(); i ++) {
   //       if(tid == 0) {
   //          D2.data[i].key = items[tid][i].first;
   //          D2.data[i].value = items[tid][i].second;
   //       }
   //       else {
   //          D2.data[partition_size[tid - 1] + i].key = items[tid][i].first;
   //          D2.data[partition_size[tid - 1] + i].value = items[tid][i].second;
   //       }
   //    }
   // }
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
