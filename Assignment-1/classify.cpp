#include "classify.h"
#include <omp.h>
#include <vector>
#include <utility>

template <typename T>
int max(std::vector<std::vector<T>> vec) {
   int max = 0;
   for(int i = 0; i < vec.size(); i ++) {
      max = (max > vec[i].size()) ? max : vec[i].size();
   }
   return max;
}

template <typename T>
int min(std::vector<std::vector<T>> vec) {
   int min = -1;
   for(int i = 0; i < vec.size(); i ++) {
      min = (min > vec[i].size() || min == -1) ? vec[i].size() : min;
   }
   return min;
}

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

// Optimisation 2: Use binary search instead of linear search
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

Data classify(Data &D, const Ranges &R, unsigned int numt)
{  
   // Classify each item in D into intervals (given by R). Finally, produce in D2 data sorted by interval
   assert(numt < MAXTHREADS);
   
   std::vector<std::vector<unsigned int>> counts(numt, std::vector<unsigned int>(R.num(), 0));
   std::vector<std::vector<int>> range(numt, std::vector<int>((int)(D.ndata / numt) + 1, 0));
   // Optimisation 4: Reduce cache line contention by allocating chunks of 8 consecutive data items to a single thread
   #pragma omp parallel num_threads(numt)
   {  
      int tid = omp_get_thread_num(); // I am thread number tid
      for(int i=tid; i<D.ndata; i+=numt) { // Threads together share-loop through all of Data
         int v = R.range_binary(D.data[i].key);// For each data, find the interval of data's key,
							  // and store the interval id in value. D is changed.
         range[tid][(int)(i / numt)] = v;
         counts[tid][v] ++;
      }
   }

   // Optimisation 3: (Minor) Compute prefix sum and accumulate sub-parts simultaneously
   // Accumulate all sub-counts (in each interval's counter) into rangecount
   unsigned int *rangecount = new unsigned int[R.num()]();
   for(int t = 0; t < numt; t ++) {
      for(int r = 1; r < R.num(); r ++) {
         rangecount[r] += counts[t][r];
      }
   }
   for(int r = 1; r < R.num(); r ++) {
      rangecount[r] += rangecount[r - 1];
   }
   // Now rangecount[i] has the number of elements in intervals before the ith interval.
   
   std::vector<std::vector<std::pair<int, int>>> partitions(numt, std::vector<std::pair<int, int>>(D.ndata / numt, {-1, -1}));
   std::vector<int> partition_size(numt, 0);
   for(int i = 0; i < D.ndata; i ++) {
      int r = range[i % numt][i / numt], p = -1;
      if(r / (R.num() / numt) >= numt)
         p = numt - 1;
      else
         p = r / (R.num() / numt);
      if(partition_size[p] == partitions[p].size())
         partitions[p].push_back({D.data[i].key, r});
      else
         partitions[p][partition_size[p]] = {D.data[i].key, r};
      partition_size[p] ++;
   }
   Data D2 = Data(D.ndata); // Make a copy
   std::vector<std::vector<std::pair<int, int>>> items(numt, std::vector<std::pair<int, int>>(0));
   for(int i = 0; i < numt; i ++) {
      items[i].resize(partition_size[i], {-1, -1});
   }
   
   #pragma omp parallel num_threads(numt)
   {
      int tid = omp_get_thread_num();
      int lower = tid * (R.num() / numt), upper = (tid + 1 == numt) ? R.num() : (tid + 1) * (R.num() / numt);
      std::vector<int> rangeIndex(upper - lower, 0);
      int temp = (lower == 0) ? 0 : rangecount[lower - 1];
      for(int i = 0; i < partition_size[tid]; i ++) {
         int r = partitions[tid][i].second;
         if(r == -1) 
            break;
         items[tid][rangecount[r - 1] - temp + rangeIndex[r - lower] ++] = {partitions[tid][i].first, r};
      } 
   }

   int globalIndex = 0, tid = 0, tIndex = 0;
   while(globalIndex < D2.ndata) {
      if(tIndex < partition_size[tid]) {
         D2.data[globalIndex].key = items[tid][tIndex].first;
         D2.data[globalIndex].value = items[tid][tIndex].second;
         tIndex ++;
         globalIndex ++;
      }
      else {
         tid ++;
         tIndex = 0;
      }
   }

   return D2;
}
