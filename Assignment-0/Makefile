CC=g++
CFLAGS=-std=c++11 -O2 -g

sources=main.cpp classify.cpp
objects=$(sources:.cpp=.o)

classify:$(objects)
	$(CC) $(CFLAGS) -fopenmp $^ -o $@

run: classify
	./classify rfile dfile 1009072 4 3

%.o: %.cpp classify.h
	$(CC) $(CFLAGS) -fopenmp -c $<

.PHONY: clean
clean:
	rm *.o classify