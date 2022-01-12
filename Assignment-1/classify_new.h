#pragma once

#include <assert.h>
#include <mutex>
#include <iostream>

#define MAXTHREADS 64 // Maximum number of threads supported
#define BADRANGE 0

struct Data;
class Ranges;

Data classify(Data &D, const Ranges &R, unsigned int numt);

// Optimization 1: Try to align counter with 0 (mod 64) address
//                 to ensure that no two counters are in the same cache line
class alignas(32) Counter { // Aligned allocation per counter. Is that enough?
                            // Keeps per-thread subcount.
   public:
      Counter(unsigned int num=MAXTHREADS) {
         _numcount = num;
         _counts = new unsigned int[num];
         assert(_counts != NULL);
         zero();
      }

      void zero() { // Initialize
         for(int i=0; i<_numcount; i++)
            _counts[i] = 0;
      }

      void increase(unsigned int id) { // If each sub-counter belongs to a thread mutual exclusion is not needed
         assert(id < _numcount);
         _counts[id]++;
      }

      void xincrease(unsigned int id) { // Safe increment
         assert(id < _numcount);
         const std::lock_guard<std::mutex> lock(cmutex);
         _counts[id]++;
      }

      unsigned int get(unsigned int id) const { // return subcounter value for specific thread
         assert(id < _numcount);
         return _counts[id];
      }

      void inspect() {
         std::cout << "Subcounts -- ";
         for(int i=0; i<_numcount; i++)
            std::cout << i << ":" << _counts[i] << " ";
         std::cout << "\n";
      }

   private:
      unsigned volatile int *_counts;
      unsigned int _numcount; // Per-thread subcounts 
      std::mutex cmutex;
};

struct Range { // Integer range

   Range(int a=1, int b=0) { // Constructor. Defaults to *bad* range
      lo = a;
      hi = b;
   }

   bool within(int val) const { // Return if val is within this range
       return(lo <= val && val <= hi);
   }

   bool strictlyin(int val) const { // Return if val is strictly inside this range
       return(lo < val && val < hi);
   }

   int lo;
   int hi; 
};

class Ranges {
   public:
      Ranges() { // Initialize with a single unreal interval
         _num = 1;
         _ranges = new Range(1, 0); // Started with this. Its not a real interval as nothing lies inside it.
      }

      Ranges& operator+=(const Range range){ // Add one more interval to this list
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
      int range_binary(int val, bool strict = false) const {
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

      int range(int val, bool strict = false) const { // Tell the range in which val lies (strict => boundary match not ok)
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

      void inspect() {
         for(int r=0; r<_num; r++) { 
            std::cout << r << "," << &_ranges[r] << ": " << _ranges[r].lo << ", " << _ranges[r].hi << "\n"; 
         }

      }

      int num() const { return _num; }

   private:
      Range *_ranges;
      int   _num;

      void set(int i, int lo, int hi) { // set the extreme values of a specific interval
         if(i < _num) {
            _ranges[i].lo = lo;
            _ranges[i].hi = hi;
         }
      }

      bool newrange(const Range r) { // Is the range r already in my list, or is it a new one?
         return (range(r.lo, true) == BADRANGE && range(r.hi, true) == BADRANGE); // Overlaps are not allowed.
      }
};

struct Data {

   struct Item {
      int key;
      int value = -1;
      Item() {
         key = 0;
         value = -1;
      }
      Item(int key, int value) {
         this->key = key;
         this->value = value;
      }
   };

   unsigned int ndata = 0;
   Item *data = NULL;

   Data(int n) { // n = Maximum number of items  storable
      ndata = n;
      data = new Item[n];
      assert(NULL != data);
   }

   void reset() {
      for(int i=0; i<ndata; i++)
         data[i].value = -1;
   }
   void inspect() {
      for(int i=0; i<ndata; i++)
         std::cout << i << ": " << data[i].key << " -- " << data[i].value <<"\n";
   }
};
