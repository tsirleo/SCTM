#define _USE_MATH_DEFINES
#include "equation.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

#define INACCURACY "INACCURACY"
#define NUMERICAL "NUMERICAL"
#define ANALITICAL "ANALITICAL"

void dump_block_to_CSV(const Grid& g, const Block& b, const VDOUB& u_local, const std::string& filename, std::string matrix_type) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file '" << filename << "' for writing." << std::endl;
        return;
    }

    for (int i = 1; i < b.Nx + 1; ++i) {
        for (int j = 1; j < b.Ny + 1; ++j) {
            for (int k = 1; k < b.Nz + 1; ++k) {
                if (matrix_type == "NUMERICAL")
                    file << u_local[b.index(i, j, k)];
                else if (matrix_type == "ANALITICAL")
                    file << u_analytical(g, (b.x_start + i - 1) * g.h_x, (b.y_start + j - 1) * g.h_y, (b.z_start + k - 1) * g.h_z, (TIME_STEPS - 1) * g.tau);
                else if (matrix_type == "INACCURACY")
                    file << fabs(u_local[b.index(i, j, k)] - u_analytical(g, (b.x_start + i - 1) * g.h_x, (b.y_start + j - 1) * g.h_y, (b.z_start + k - 1) * g.h_z, g.tau * (TIME_STEPS - 1)));
                else
                    std::cerr << "Unknown matrix type: " << matrix_type << std::endl;

                if (k < b.Nz)
                    file << ",";
            }
            file << "\n";
        }
        file << "\n";
    }
    file.close();
}

void save_statistics(const Grid& g, double& time, double& max_inaccuracy, double& first_step_inaccuracy, double& last_step_inaccuracy, int& proc_num) {
    std::string filename = "results/statistics/" + std::to_string(g.N) + "_" + std::to_string(proc_num) + "_" + g.L_type + "_" + "statistics.txt";
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file '" << filename << "' for writing." << std::endl;
        return;
    }

    file << "Time = " << time << "\n" << "Max inaccuracy = " << max_inaccuracy << "\n" << "First step inaccuracy = " << first_step_inaccuracy << "\n" << "Last step inaccuracy = " << last_step_inaccuracy << std::endl;
}

int get_0_dim_size(const int& proc_num) {
    if (proc_num > 10 and proc_num % 4 == 0)
        return 4;
    else if (proc_num % 2 == 0)
        return 2;
    else
        return 1;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, proc_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

    if ((!strcmp(argv[2], "custom") and argc != 6) or (strcmp(argv[2], "custom") and argc != 3)) {
        if (rank == 0)
            std::cerr << "Invalid number of arguments. You must specify 3 or 6." << "\n" << "Syntaxis: N,  L_type (1 -> Lx=Ly=Lz=1, pi -> Lx=Ly=Lz=Pi, custom -> specify 3 extra values for Lx, Ly, Lz), Lx, Ly, Lz" << std::endl;

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int N = atoi(argv[1]);
    if (N < 0) {
        if (rank == 0)
            std::cerr << "Invalid N: must be > 0" << std::endl;

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    Grid grid = Grid();
    if (!strcmp(argv[2], "pi"))
        grid = Grid(N, argv[2], M_PI);
    else if (!strcmp(argv[2], "1"))
        grid = Grid(N, argv[2], 1);
    else if (!strcmp(argv[2], "custom"))
        grid = Grid(N, argv[2], atof(argv[3]), atof(argv[4]), atof(argv[5]));
    else {
        if (rank == 0)
            std::cerr << "Invalid L_type: must be '1', 'pi' or 'custom'" << std::endl;

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0)
        std::cout << "Input values: \n\t" << "N = " << grid.N << "\n\t" << "Processes = " << proc_num << "\n\t" << "L_type = " << grid.L_type <<
            "\n\t" << "Lx = " << grid.Lx << "\n\t" << "Ly = " << grid.Ly << "\n\t" << "Lz = " << grid.Lz << std::endl;

    // create cartesian topology
    int dims[3] = {get_0_dim_size(proc_num), 0, 0};
    MPI_Dims_create(proc_num, 3, dims);
    if (rank == 0)
        std::cout << "Dims topology: (" << dims[0] << ", " << dims[1] << ", " << dims[2] << ")" << std::endl;

    int periods[3] = {1, 1, 1}; // periodic boundaries
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &comm_cart);

    int coords[3];
    MPI_Cart_coords(comm_cart, rank, 3, coords);

    // neighbors in 6 directions
    VINT neighbors(6);
    MPI_Cart_shift(comm_cart, 0, 1, &neighbors[0], &neighbors[1]); // (left and right)
    MPI_Cart_shift(comm_cart, 1, 1, &neighbors[2], &neighbors[3]); // (bottom and top)
    MPI_Cart_shift(comm_cart, 2, 1, &neighbors[4], &neighbors[5]); // (front and back)

    VDOUB result_vec;
    double time, first_step_inaccuracy, last_step_inaccuracy, max_inaccuracy = -1;
    Block block = Block(grid, neighbors, coords, dims[0], dims[1], dims[2], rank);
//    block.print_block_info();
    solve_equation(grid, block, dims[0], dims[1], dims[2], comm_cart, time, max_inaccuracy, first_step_inaccuracy, last_step_inaccuracy, result_vec);

    if (rank == 0) {
        std::cout << "Result: \n\t" << "Time = " << time << "\n\t" << "Max inaccuracy = " << max_inaccuracy << "\n\t" << "First step inaccuracy = " << first_step_inaccuracy << "\n\t" << "Last step inaccuracy = " << last_step_inaccuracy << std::endl;
        save_statistics(grid, time, max_inaccuracy, first_step_inaccuracy, last_step_inaccuracy, proc_num);
    }

//    dump_block_to_CSV(grid, block, result_vec, "results/grid/" + std::to_string(N) + "_" + std::to_string(proc_num) + "_" + grid.L_type + "_" + NUMERICAL + std::to_string(block.dim0_rank) + std::to_string(block.dim1_rank) + std::to_string(block.dim2_rank) + ".csv", NUMERICAL);
//    dump_block_to_CSV(grid, block, result_vec, "results/grid/" + std::to_string(N) + "_" + std::to_string(proc_num) + "_" + grid.L_type + "_" + ANALITICAL + std::to_string(block.dim0_rank) + std::to_string(block.dim1_rank) + std::to_string(block.dim2_rank) +  ".csv", ANALITICAL);
//    dump_block_to_CSV(grid, block, result_vec, "results/grid/" + std::to_string(N) + "_" + std::to_string(proc_num) + "_" + grid.L_type + "_" + INACCURACY + std::to_string(block.dim0_rank) + std::to_string(block.dim1_rank) + std::to_string(block.dim2_rank) +  ".csv", INACCURACY);

    MPI_Finalize();
    return 0;
}
