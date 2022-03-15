a.out: *.cpp *.hpp
	mpic++ -fopenmp -g --std=c++17 main.cpp hnsw.cpp util.cpp
run: a.out
	mpirun --bind-to none ./a.out out_data 15 abc abc
setup: setup.cpp
	g++ -std=c++17 setup.cpp -o setup
run_setup: setup
	./setup out_data out_data
