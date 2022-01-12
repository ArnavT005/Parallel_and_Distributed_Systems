CC=g++
#g++ -std=c++11 -O2 -mavx2 -c classify.cpp 
CFLAGS=-std=c++11 -O2

sources=main.cpp classify.cpp
sources_new=main.cpp classify_new.cpp
objects=$(sources:.cpp=.o)
objects_new=$(sources_new:.cpp=.o)

classify:$(objects)
	$(CC) $(CFLAGS) -fopenmp $^ -o $@

classify_new:$(objects_new)
	$(CC) $(CFLAGS) -fopenmp $^ -o $@

run: classify
	./classify rfile dfile 1009072 8 3

run_new: classify_new
	perf stat ./classify_new rfile dfile 1009072 8 3

#classify.o: classify.h
#	touch classify.o
#	$(CC) $(CFLAGS) -c $<
#main.o: classify.h
#	$(CC) $(CFLAGS) -c $<

%.o: %.cpp classify.h classify_new.h
	$(CC) $(CFLAGS) -c -fopenmp $<

.PHONY: clean
clean:
	rm *.o classify classify_new
