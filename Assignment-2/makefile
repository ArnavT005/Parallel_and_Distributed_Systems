compile:
	mpic++ -std=c++17 -o wtf main.cpp librandomizer.a
run:
	time mpirun -np 4 ./wtf data/81867_545671/edges.dat 81867 545671 0.1 100 30 20 369
clean:
	rm wtf