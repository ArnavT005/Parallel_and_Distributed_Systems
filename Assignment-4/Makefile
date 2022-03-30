main: main.cu img.hpp
	nvcc -o main main.cu img.cpp

run: main
	./main data_image.txt query_image.txt 1.2 1.2 5

clean:
	rm -f main