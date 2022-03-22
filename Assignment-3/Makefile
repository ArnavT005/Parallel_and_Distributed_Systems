N?=2
C?=5
K?=10
a.out: *.cpp *.hpp
	mpic++ -Ofast -fopenmp -g --std=c++17 main.cpp hnsw.cpp util.cpp
run: a.out
	/usr/bin/time -v mpirun --bind-to none ./a.out out_data $K abc out_data/preds/$N_$C_$K.txt 2>&1
setup: setup.cpp
	mpic++ -fopenmp -std=c++17 setup.cpp -o setup
run_setup: setup
	./setup out_data out_data
clean:
	-rm a.out
	-rm setup
