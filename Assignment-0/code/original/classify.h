#pragma once

#include <assert.h>
#include <mutex>
#include <iostream>

#define MAXTHREADS 64 // Maximum number of threads supported
#define BADRANGE 0

struct Data;
class Ranges;

Data classify(Data &D, const Ranges &R, unsigned int numt);

class alignas(32) Counter { // Aligned allocation per counter. Is that enough?
                            // Keeps per-thread subcount.
   public:
      
      Counter(unsigned int num);

      void zero();

      void increase(unsigned int id);

      void xincrease(unsigned int id);

      unsigned int get(unsigned int id) const;

      void inspect();
   private:
      unsigned volatile int *_counts;
      unsigned int _numcount; // Per-thread subcounts 
      std::mutex cmutex;
};

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

      int range(int val, bool strict) const;

      void inspect();
      int num() const;

   private:
      Range *_ranges;
      int   _num;
      
      void set(int i, int lo, int hi);

      bool newrange(const Range r);
};

struct Data {

   struct Item {
      int key;
      int value = -1;
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
