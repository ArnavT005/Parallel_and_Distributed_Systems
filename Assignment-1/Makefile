CC=g++
CFLAGS=-std=c++11 -O2

sources=main.cpp classify.cpp
sources_new=main.cpp classify_new.cpp
objects=$(sources:.cpp=.o)
objects_new=$(sources_new:.cpp=.o)

classify:$(objects)
	$(CC) $(CFLAGS) -g -fopenmp $^ -o $@

run: classify
	./classify rfile dfile 1009072 63 1

%.o: %.cpp classify.h
	$(CC) $(CFLAGS) -g -fopenmp -c $<

.PHONY: clean
clean:
	rm *.o classify