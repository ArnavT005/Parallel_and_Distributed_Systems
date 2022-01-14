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

struct Range { // Integer range

   Range(int a, int b);

   bool within(int val) const;

   bool strictlyin(int val) const;

   int lo;
   int hi; 
};

class Ranges {
   public:
      Ranges();
      Ranges& operator+=(const Range range);

      // Optimisation 2: Use binary search instead of linear search
      int range_binary(int val, bool strict) const;

      int range(int val, bool strict) const;

      void inspect();

      int num() const;

   private:
      Range *_ranges;
      int   _num;

      void set(int i, int lo, int hi);

      bool newrange(const Range r);
};

struct Item {
   int key;
   int value;
   Item() {
      key = 0;
      value = -1;
   }
   Item(int key, int value) {
      this->key = key;
      this->value = value;
   }
};

struct Data {

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
