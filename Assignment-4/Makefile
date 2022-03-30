CFLAGS := --std=c++14
main: *.cu *.cuh *.cpp *.hpp
	nvcc -dc $(CFLAGS)  main.cu -o main.o
	nvcc -dc $(CFLAGS) util.cu -o util.o
	nvcc -dc $(CFLAGS) img.cpp -o img.o
	nvcc $(CFLAGS) main.o util.o img.o -o a.out
run: a.out
	CUDA_LAUNCH_BLOCKING=1 ./a.out data_image.txt query_image.txt 1.2 1.2 5

clean:
	rm -f *.o a.out