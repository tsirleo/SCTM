#define _USE_MATH_DEFINES
#include "equation.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>

#define INACCURACY "INACCURACY"
#define NUMERICAL "NUMERICAL"
#define ANALITICAL "ANALITICAL"

void dump_grid_to_CSV(const Grid& g, const VDOUB& u, const std::string& filename, std::string matrix_type) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file '" << filename << "' for writing." << std::endl;
        return;
    }

    for (int i = 0; i < g.N + 1; ++i) {
        for (int j = 0; j < g.N + 1; ++j) {
            for (int k = 0; k < g.N + 1; ++k) {
                if (matrix_type == "NUMERICAL")
                    file << u[g.index(i, j, k)];
                else if (matrix_type == "ANALITICAL")
                    file << u_analytical(g, i * g.h_x, j * g.h_y, k * g.h_z, (TIME_STEPS - 1) * g.tau);
                else if (matrix_type == "INACCURACY")
                    file << fabs(u[g.index(i, j, k)] - u_analytical(g, i * g.h_x, j * g.h_y, k * g.h_z, (TIME_STEPS - 1) * g.tau));
                else
                    std::cerr << "Unknown matrix type: " << matrix_type << std::endl;
                if (k < g.N)
                    file << ",";
            }
            file << "\n";
        }
        file << "\n";
    }
    file.close();
}

void save_statistics(const Grid& g, double& time, double& max_inaccuracy, double& first_step_inaccuracy, double& last_step_inaccuracy, int& threads_num) {
    std::string filename = "results/statistics/" + std::to_string(g.N) + "_" + std::to_string(threads_num) + "_" + g.L_type + "_" + "statistics.txt";
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file '" << filename << "' for writing." << std::endl;
        return;
    }

    file << "Time = " << time << "\n" << "Max inaccuracy = " << max_inaccuracy << "\n" << "First step inaccuracy = " << first_step_inaccuracy << "\n" << "Last step inaccuracy = " << last_step_inaccuracy << std::endl;
}

int main(int argc, char* argv[]) {
    if ((!strcmp(argv[3], "custom") and argc != 7) or (strcmp(argv[3], "custom") and argc != 4)) {
        std::cerr << "Invalid number of arguments. You must specify 3 or 6." << "\n" << "Syntaxis: N, Treads,  L_type (1 -> Lx=Ly=Lz=1, pi -> Lx=Ly=Lz=Pi, custom -> specify 3 extra values for Lx, Ly, Lz), Lx, Ly, Lz" << std::endl;
        return 0;
    }

    int N = atoi(argv[1]);
    if (N < 0) {
        std::cerr << "Invalid N: must be > 0" << std::endl;
        return 0;
    }

    int threads_num = atoi(argv[2]);
    if (threads_num < 0) {
        std::cerr << "Invalid Treads number: must be > 0" << std::endl;
        return 0;
    }

    Grid grid = Grid();
    if (!strcmp(argv[3], "pi"))
        grid = Grid(N, argv[3], M_PI);
    else if (!strcmp(argv[3], "1"))
        grid = Grid(N, argv[3], 1);
    else if (!strcmp(argv[3], "custom"))
        grid = Grid(N, argv[3], atof(argv[4]), atof(argv[5]), atof(argv[6]));
    else {
        std::cerr << "Invalid L_type: must be '1', 'pi' or 'custom'" << std::endl;
        return 0;
    }

    std::cout << "Input values: \n\t" << "N = " << grid.N << "\n\t" << "Treads = " << threads_num << "\n\t" << "L_type = " << grid.L_type <<
        "\n\t" << "Lx = " << grid.Lx << "\n\t" << "Ly = " << grid.Ly << "\n\t" << "Lz = " << grid.Lz << std::endl;

    VDOUB result_vec;
    double time, first_step_inaccuracy, last_step_inaccuracy, max_inaccuracy = -1;
    solve_equation(grid, time, max_inaccuracy, first_step_inaccuracy, last_step_inaccuracy, result_vec, threads_num);

    std::cout << "Result: \n\t" << "Time = " << time << "\n\t" << "Max inaccuracy = " << max_inaccuracy << "\n\t" << "First step inaccuracy = " << first_step_inaccuracy << "\n\t" << "Last step inaccuracy = " << last_step_inaccuracy << std::endl;
    save_statistics(grid, time, max_inaccuracy, first_step_inaccuracy, last_step_inaccuracy, threads_num);

    dump_grid_to_CSV(grid, result_vec, "results/grid/" + std::to_string(N) + "_" + std::to_string(threads_num) + "_" + grid.L_type + "_" + NUMERICAL + ".csv", NUMERICAL);
    dump_grid_to_CSV(grid, result_vec, "results/grid/" + std::to_string(N) + "_" + std::to_string(threads_num) + "_" + grid.L_type + "_" + ANALITICAL + ".csv", ANALITICAL);
    dump_grid_to_CSV(grid, result_vec, "results/grid/" + std::to_string(N) + "_" + std::to_string(threads_num) + "_" + grid.L_type + "_" + INACCURACY + ".csv", INACCURACY);

    return 0;
}
