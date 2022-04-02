CFLAGS := --std=c++11

main: *.cu *.cuh *.cpp *.hpp
	nvcc -dc $(CFLAGS)  main.cu -o main.o
	nvcc -dc $(CFLAGS) util.cu -o util.o
	nvcc -dc $(CFLAGS) img.cpp -o img.o
	nvcc $(CFLAGS) main.o util.o img.o -o main

run: main
	./main data_image.txt query_image.txt 1000 2 10

clean:
	rm -f *.o main output.txt