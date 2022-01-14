#include "classify.h"
#include <omp.h>

Counter::Counter(unsigned int num=MAXTHREADS) {
   _numcount = num;
   _counts = new unsigned int[num];
   assert(_counts != NULL);
   zero();
}

void Counter::zero() { // Initialize
   for(int i=0; i<_numcount; i++)
      _counts[i] = 0;
}

void Counter::increase(unsigned int id) { // If each sub-counter belongs to a thread mutual exclusion is not needed
   assert(id < _numcount);
   _counts[id]++;
}

void Counter::xincrease(unsigned int id) { // Safe increment
   assert(id < _numcount);
   const std::lock_guard<std::mutex> lock(cmutex);
   _counts[id]++;
}

unsigned int Counter::get(unsigned int id) const { // return subcounter value for specific thread
   assert(id < _numcount);
   return _counts[id];
}

void Counter::inspect() {
   std::cout << "Subcounts -- ";
   for(int i=0; i<_numcount; i++)
      std::cout << i << ":" << _counts[i] << " ";
   std::cout << "\n";
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

void Ranges::set(int i, int lo, int hi) { // set the extreme values of a specific interval
   if(i < _num) {
      _ranges[i].lo = lo;
      _ranges[i].hi = hi;
   }
}

bool Ranges::newrange(const Range r) { // Is the range r already in my list, or is it a new one?
   return (range(r.lo, true) == BADRANGE && range(r.hi, true) == BADRANGE); // Overlaps are not allowed.
}

Data classify(Data &D, const Ranges &R, unsigned int numt)
{  
   // Classify each item in D into intervals (given by R). Finally, produce in D2 data sorted by interval
   assert(numt < MAXTHREADS);
   Counter counts[R.num()]; // I need on counter per interval. Each counter can keep pre-thread subcount.
   #pragma omp parallel num_threads(numt)
   {
      int tid = omp_get_thread_num(); // I am thread number tid
      for(int i=tid; i<D.ndata; i+=numt) { // Threads together share-loop through all of Data
         int v = D.data[i].value = R.range(D.data[i].key);// For each data, find the interval of data's key,
							  // and store the interval id in value. D is changed.
         counts[v].increase(tid); // Found one key in interval v
      }
   }

   // Accumulate all sub-counts (in each interval;'s counter) into rangecount
   unsigned int *rangecount = new unsigned int[R.num()];
   for(int r=0; r<R.num(); r++) { // For all intervals
      rangecount[r] = 0;
      for(int t=0; t<numt; t++) // For all threads
         rangecount[r] += counts[r].get(t);
      // std::cout << rangecount[r] << " elements in Range " << r << "\n"; // Debugging statement
   }

   // Compute prefx sum on rangecount.
   for(int i=1; i<R.num(); i++) {
      rangecount[i] += rangecount[i-1];
   }

   // Now rangecount[i] has the number of elements in intervals before the ith interval.

   Data D2 = Data(D.ndata); // Make a copy
   
   #pragma omp parallel num_threads(numt)
   {
      int tid = omp_get_thread_num();
      for(int r=tid; r<R.num(); r+=numt) { // Thread together share-loop through the intervals 
         int rcount = 0;
         for(int d=0; d<D.ndata; d++) // For each interval, thread loops through all of data and  
             if(D.data[d].value == r) // If the data item is in this interval 
                 D2.data[rangecount[r-1]+rcount++] = D.data[d]; // Copy it to the appropriate place in D2.
      }
   }

   return D2;
}
