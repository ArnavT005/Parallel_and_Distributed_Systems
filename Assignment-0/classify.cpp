#include "classify.h"
#include <omp.h>
#include <vector>

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

Data classify(Data &D, const Ranges &R, unsigned int numt)
{  
   assert(numt < MAXTHREADS);
   
   std::vector<std::vector<unsigned int>> counts(numt, std::vector<unsigned int>(R.num(), 0));
   int k = 12;
   #pragma omp parallel num_threads(numt)
   {
      int tid = omp_get_thread_num(); // I am thread number tid
      int index = k * tid;
      while(index < D.ndata) {
         for(int j = index; j < (index + k) && j < D.ndata; j ++) {
            int v = D.data[j].value = R.range(D.data[j].key);
            counts[tid][v] ++;
         }
         index += k * numt;
      }
   }

   unsigned int *rangecount = new unsigned int[R.num()]();
   int p = -1;
   for(int t = 0; t < numt; t ++) {
      for(int r = 1; r < R.num(); r ++) {
         rangecount[r] += counts[t][r];   
      }
   }
   for(int r = 1; r < R.num(); r ++) {
      rangecount[r] += rangecount[r - 1];
   }
   
   Data D2 = Data(D.ndata);
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
   return D2;
}
