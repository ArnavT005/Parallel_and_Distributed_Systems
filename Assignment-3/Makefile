a.out: main.cpp
	mpic++ --std=c++17 main.cpp
run: a.out
	./a.out data 10 abc abc
setup: setup.cpp
	g++ -std=c++17 setup.cpp -o setup
run_setup: setup
	./setup data out_data