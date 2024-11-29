CFLAGS = -O3 -Wall -Wextra -std=c++11 -fopenmp
CC = clang++ -I/opt/homebrew/Cellar/open-mpi/5.0.5/include
CCEXTRA = -I/opt/homebrew/Cellar/open-mpi/5.0.5/include -L/opt/homebrew/Cellar/open-mpi/5.0.5/lib -lmpi
CC_POL = OMPI_CXX=g++ mpicxx -I/opt/ibm/spectrum_mpi/include
CCEXTRA_POL = -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm

compile: create_result_dir
	$(CC) $(CCEXTRA) $(CFLAGS) equation_solution.cpp main.cpp -o equation

compile_polus: create_result_dir
	$(CC_POL) $(CCEXTRA_POL) $(CFLAGS) equation_solution.cpp main.cpp -o equation

create_result_dir:
	mkdir -p results/statistics
	mkdir -p results/grid

run_64_1: compile
	for procs in 1 2 4 8 16 32 ; do \
		for threads in 1 2 4 8 ; do \
			mpirun -np $$procs -map-by :OVERSUBSCRIBE ./equation 64 $$threads 1 ; \
		done \
	done \

run_64_pi: compile
	for procs in 1 2 4 8 16 32 ; do \
		for threads in 1 2 4 8 ; do \
			mpirun -np $$procs -map-by :OVERSUBSCRIBE ./equation 64 $$threads pi ; \
		done \
	done \

run_128_1: compile
	for procs in 1 2 4 8 16 32 ; do \
		for threads in 1 2 4 8 ; do \
			mpirun -np $$procs -map-by :OVERSUBSCRIBE ./equation 128 $$threads 1 ; \
		done \
	done \

run_128_pi: compile
	for procs in 1 2 4 8 16 32 ; do \
		for threads in 1 2 4 8 ; do \
			mpirun -np $$procs -map-by :OVERSUBSCRIBE ./equation 128 $$threads pi ; \
		done \
	done \

run_256_1: compile
	for procs in 1 2 4 8 16 32 ; do \
		for threads in 1 2 4 8 ; do \
			mpirun -np $$procs -map-by :OVERSUBSCRIBE ./equation 256 $$threads 1 ; \
		done \
	done \

run_256_pi: compile
	for procs in 1 2 4 8 16 32 ; do \
		for threads in 1 2 4 8 ; do \
			mpirun -np $$procs -map-by :OVERSUBSCRIBE ./equation 256 $$threads pi ; \
		done \
	done \

run_512_1: compile
	for procs in 1 2 4 8 16 32 ; do \
		for threads in 1 2 4 8 ; do \
			mpirun -np $$procs -map-by :OVERSUBSCRIBE ./equation 512 $$threads 1 ; \
		done \
	done \

run_512_pi: compile
	for procs in 1 2 4 8 16 32 ; do \
		for threads in 1 2 4 8 ; do \
			mpirun -np $$procs -map-by :OVERSUBSCRIBE ./equation 512 $$threads pi ; \
		done \
	done \

run_all_local: compile
	for N in 128 256 512 ; do \
		for type in '1' 'pi' ; do \
		  	for procs in 1 2 4 8 16 32 ; do \
				for threads in 1 2 4 8 ; do \
					mpirun -np $$procs -map-by :OVERSUBSCRIBE ./equation $$N $$threads $$type ; \
				done \
			done \
		done \
	done \

run_all_polus: compile_polus
	for N in 128 256 512 ; do \
		for type in '1' 'pi' ; do \
		  	for procs in 1 2 4 8 16 32 ; do \
				for threads in 1 2 4 8 ; do \
					mpisubmit.pl -p $$procs -t $$threads -w 00:30 --stdout stats\_$$N\_$$procs\_$$threads\_$$type.out --stderr stats\_$$N\_$$procs\_$$threads\_$$type.err ./equation $$N $$threads $$type ; \
				done \
			done \
		done \
	done \


clean_results:
	rm -rf results/*

clean:
	rm -f *.o equation

clean_all:
	rm -rf results/*
	rm -r results
	rm -f *.o equation