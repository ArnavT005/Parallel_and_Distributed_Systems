CFLAGS := --std=c++11

main: *.cu *.cuh *.cpp *.hpp
	nvcc -dc $(CFLAGS)  main.cu -o main.o
	nvcc -dc $(CFLAGS) util.cu -o util.o
	nvcc -dc $(CFLAGS) img.cpp -o img.o
	nvcc $(CFLAGS) main.o util.o img.o -o main

run: main
	./main base_case/t4/data_image_a.txt base_case/t4/query_image_a_.txt 12 7.5 20

clean:
	rm -f *.o main output.txt
