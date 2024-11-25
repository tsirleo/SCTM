CFLAGS = -O3 -Wall -Wextra -std=c++11
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
	./equation 64 1
run_64_pi: compile
	./equation 64 pi
run_128_1: compile
	./equation 128 1
run_128_pi: compile
	./equation 128 pi
run_256_1: compile
	./equation 256 1
run_256_pi: compile
	./equation 256 pi
run_512_1: compile
	./equation 512 1
run_512_pi: compile
	./equation 512 pi

run_all_local: compile
	for N in 64 128 256 512 ; do \
		for type in '1' 'pi' ; do \
			./equation $$N $$type ; \
		done \
	done \

run_all_polus: compile
	for N in 128 256 512 ; do \
		for type in '1' 'pi' ; do \
			bsub -n 1 -q short -W 00:30 -J sequential_job\_$$N\_$$type -o stats\_$$N\_$$type.out -e stats\_$$N\_$$type.err -R "affinity[core(10,same=socket,exclusive=(socket,alljobs)):membind=localonly:distribute=pack(socket=1)]"  ./equation $$N $$type ; \
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