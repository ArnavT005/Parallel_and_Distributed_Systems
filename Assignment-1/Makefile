CC=g++
CFLAGS=-std=c++11 -O2 -g

sources=driver.cpp psort.cpp
objects=$(sources:.cpp=.o)

all:
	g++ -fPIC -g -c -Wall psort.cpp -fopenmp
	g++ -shared -Wl,-soname,libsort.so -o libsort.so psort.o -lc

main:$(objects)
	$(CC) $(CFLAGS) -fopenmp $^ -o $@

%.o: %.cpp psort.h
	$(CC) $(CFLAGS) -fopenmp -c $<

.PHONY: clean
clean:
	rm *.so *.o