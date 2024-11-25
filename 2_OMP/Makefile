CFLAGS = -O3 -Wall -Wextra -std=c++11 -fopenmp
CC = clang++
CC_POL = g++

compile: create_result_dir
	$(CC) $(CFLAGS) equation_solution.cpp main.cpp -o equation

compile_polus: create_result_dir
	$(CC_POL) $(CFLAGS) equation_solution.cpp main.cpp -o equation

create_result_dir:
	mkdir -p results/statistics
	mkdir -p results/grid

run_64_1: compile
	for threads in 1 2 4 8 16 32 ; do \
		./equation 64 $$threads 1 ; \
	done \

run_64_pi: compile
	for threads in 1 2 4 8 16 32 ; do \
		./equation 64 $$threads pi ; \
	done \

run_128_1: compile
	for threads in 1 2 4 8 16 32 ; do \
		./equation 128 $$threads 1 ; \
	done \

run_128_pi: compile
	for threads in 1 2 4 8 16 32 ; do \
		./equation 128 $$threads pi ; \
	done \

run_256_1: compile
	for threads in 1 2 4 8 16 32 ; do \
		./equation 256 $$threads 1 ; \
	done \

run_256_pi: compile
	for threads in 1 2 4 8 16 32 ; do \
		./equation 256 $$threads pi ; \
	done \

run_512_1: compile
	for threads in 1 2 4 8 16 32 ; do \
		./equation 512 $$threads 1 ; \
	done \

run_512_pi: compile
	for threads in 1 2 4 8 16 32 ; do \
		./equation 512 $$threads pi ; \
	done \

run_all_local: compile
	for N in 128 256 512 ; do \
		for type in '1' 'pi' ; do \
		  	for threads in 1 2 4 8 16 32 ; do \
				./equation $$N $$threads $$type ; \
			done \
		done \
	done \

run_all_polus: compile_polus
	for N in 128 256 512 ; do \
		for type in '1' 'pi' ; do \
		  	for threads in 1 2 4 8 16 32 ; do \
				bsub -n 1 -q short -W 00:30 -J omp_job\_$$N\_$$threads\_$$type -o stats\_$$N\_$$threads\_$$type.out -e stats\_$$N\_$$threads\_$$type.err -R "affinity[core(10,same=socket,exclusive=(socket,alljobs)):membind=localonly:distribute=pack(socket=1)]"  ./equation $$N $$threads $$type ; \
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