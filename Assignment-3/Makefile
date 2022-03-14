a.out: *.cpp *.hpp
	mpic++ -g --std=c++17 main.cpp hnsw.cpp util.cpp
run: a.out
	mpirun -np 1 ./a.out out_data 10 abc abc
setup: setup.cpp
	g++ -std=c++17 setup.cpp -o setup
run_setup: setup
	./setup out_data out_data
