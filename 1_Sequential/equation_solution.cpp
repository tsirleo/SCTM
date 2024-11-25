#define _USE_MATH_DEFINES
#include "equation.h"
#include <iostream>
#include <chrono>
#include <cmath>

inline double laplace_operator(const Grid& g, const VDOUB& ui, const int& i, const int& j, const int& k) {
    return (ui[g.index((i - 1) < 0 ? g.N - 1 : i - 1, j, k)] - 2 * ui[g.index(i, j, k)] + ui[g.index((i + 1) > g.N ? 1 : i + 1, j, k)]) / pow(g.h_x, 2) +
           (ui[g.index(i, (j - 1) < 0 ? g.N - 1 : j - 1, k)] - 2 * ui[g.index(i, j, k)] + ui[g.index(i, (j + 1) > g.N ? 1 : j + 1, k)]) / pow(g.h_y, 2) +
           (ui[g.index(i, j, (k - 1) < 0 ? g.N - 1 : k - 1)] - 2 * ui[g.index(i, j, k)] + ui[g.index(i, j, (k + 1) > g.N ? 1 : k + 1)]) / pow(g.h_z, 2);
}

void init(const Grid& g, VVEC& u, double& max_inaccuracy, double& first_step_inaccuracy) {
    // boundary conditions (7-9) variant 8: x - П, y - П, z - П
    for (int i = 0; i < g.N + 1; ++i) {
        for (int j = 0; j < g.N + 1; ++j) {
            u[0][g.index(0, i, j)] = u_analytical(g, 0, i * g.h_y, j * g.h_z, 0);
            u[0][g.index(g.N, i, j)] = u_analytical(g, g.Lx, i * g.h_y, j * g.h_z, 0);

            u[1][g.index(0, i, j)] = u_analytical(g, 0, i * g.h_y, j * g.h_z, g.tau);
            u[1][g.index(g.N, i, j)] = u_analytical(g, g.Lx, i * g.h_y, j * g.h_z, g.tau);
        }
    }

    for (int i = 0; i < g.N + 1; ++i) {
        for (int j = 0; j < g.N + 1; ++j) {
            u[0][g.index(i, 0, j)] = u_analytical(g, i * g.h_x, 0, j * g.h_z, 0);
            u[0][g.index(i, g.N, j)] = u_analytical(g, i * g.h_x, g.Ly, j * g.h_z, 0);

            u[1][g.index(i, 0, j)] = u_analytical(g, i * g.h_x, 0, j * g.h_z, g.tau);
            u[1][g.index(i, g.N, j)] = u_analytical(g, i * g.h_x, g.Ly, j * g.h_z, g.tau);
        }
    }

    for (int i = 0; i < g.N + 1; ++i) {
        for (int j = 0; j < g.N + 1; ++j) {
            u[0][g.index(i, j, 0)] = u_analytical(g, i * g.h_x, j * g.h_y, 0, 0);
            u[0][g.index(i, j, g.N)] = u_analytical(g, i * g.h_x, j * g.h_y, g.Lz, 0);

            u[1][g.index(i, j, 0)] = u_analytical(g, i * g.h_x, j * g.h_y, 0, g.tau);
            u[1][g.index(i, j, g.N)] = u_analytical(g, i * g.h_x, j * g.h_y, g.Lz, g.tau);
        }
    }

	// u_0 (10)
    for (int i = 1; i < g.N; ++i)
    	for (int j = 1; j < g.N; ++j)
      		for (int k = 1; k < g.N; ++k)
        		u[0][g.index(i, j, k)] = u_analytical(g, i * g.h_x, j * g.h_y, k * g.h_z, 0);

	// u_1 (12)
    for (int i = 1; i < g.N; ++i)
    	for (int j = 1; j < g.N; ++j)
      		for (int k = 1; k < g.N; ++k)
       			 u[1][g.index(i, j, k)] = u[0][g.index(i, j, k)] + 0.5 * pow(g.tau, 2) * laplace_operator(g, u[0], i, j, k);

    // innacuracy calculation
    double step_max_error = -1;
    for (int i = 0; i < g.N + 1; ++i) {
        for (int j = 0; j < g.N + 1; ++j) {
            for (int k = 0; k < g.N + 1; ++k) {
                double tmp = fabs(u[1][g.index(i, j, k)] - u_analytical(g, i * g.h_x, j * g.h_y, k * g.h_z, g.tau));

                if (tmp > step_max_error)
                    step_max_error = tmp;
            }
        }
    }

    if (step_max_error > max_inaccuracy)
        max_inaccuracy = step_max_error;

    first_step_inaccuracy = step_max_error;

    std::cout << "Steps inaccuracy:\n\t" << "Max inaccuracy on step 1" << " = " << step_max_error << std::endl;
}

void run_algo(Grid& g, VVEC& u, double& max_inaccuracy, double& last_step_inaccuracy) {
    int next, curr, prev;
    for (int s = 2; s < TIME_STEPS; ++s) {
        next = s % 3;
        curr = (s - 1) % 3;
        prev = (s - 2) % 3;

        // boundary conditions
        for (int i = 0; i < g.N + 1; ++i) {
            for (int j = 0; j < g.N + 1; ++j) {
                double value = 2 * u[curr][g.index(0, i, j)] - u[prev][g.index(0, i, j)] + pow(g.tau, 2) * laplace_operator(g, u[curr], 0, i, j);
                u[next][g.index(0, i, j)] = value;
                u[next][g.index(g.N, i, j)] = value;
            }
        }

        for (int i = 1; i < g.N; ++i) {
            for (int j = 0; j < g.N + 1; ++j) {
                double value = 2 * u[curr][g.index(i, 0, j)] - u[prev][g.index(i, 0, j)] + pow(g.tau, 2) * laplace_operator(g, u[curr], i, 0, j);
                u[next][g.index(i, 0, j)] = value;
                u[next][g.index(i, g.N, j)] = value;
            }
        }

        for (int i = 1; i < g.N; ++i) {
            for (int j = 1; j < g.N; ++j) {
                double value = 2 * u[curr][g.index(i, j, 0)] - u[prev][g.index(i, j, 0)] + pow(g.tau, 2) * laplace_operator(g, u[curr], i, j, 0);
                u[next][g.index(i, j, 0)] = value;
                u[next][g.index(i, j, g.N)] = value;
            }
        }

        // u_n
        for (int i = 1; i < g.N; ++i) {
            for (int j = 1; j < g.N; ++j) {
                for (int k = 1; k < g.N; ++k) {
                    u[next][g.index(i, j, k)] = 2 * u[curr][g.index(i, j, k)] - u[prev][g.index(i, j, k)] + pow(g.tau, 2) * laplace_operator(g, u[curr], i, j, k);
                }
            }
        }

        // innacuracy calculation
        double step_max_error = -1;
        for (int i = 0; i < g.N + 1; ++i) {
            for (int j = 0; j < g.N + 1; ++j) {
                for (int k = 0; k < g.N + 1; ++k) {
                    double tmp = fabs(u[next][g.index(i, j, k)] - u_analytical(g, i * g.h_x, j * g.h_y, k * g.h_z, g.tau * s));

                    if (tmp > step_max_error)
                        step_max_error = tmp;
                }
            }
        }

        if (step_max_error > max_inaccuracy)
            max_inaccuracy = step_max_error;

        if (s == TIME_STEPS - 1)
            last_step_inaccuracy = step_max_error;

        std::cout << "\tMax inaccuracy on step " << s << " = " << step_max_error << std::endl;
    }
}

void solve_equation(Grid& g, double& time, double& max_inaccuracy, double& first_step_inaccuracy, double& last_step_inaccuracy, VDOUB& result) {
    int n = pow(g.N + 1, 3);
    VDOUB u0(n), u1(n), u2(n);
    VVEC u{u0, u1, u2};

    auto start = std::chrono::high_resolution_clock::now();

    init(g, u, max_inaccuracy, first_step_inaccuracy);
    run_algo(g, u, max_inaccuracy, last_step_inaccuracy);

    auto end = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double, std::milli> fp_ms = end - start;
    time = fp_ms.count() / 1000;

    result = u[(TIME_STEPS - 1) % 3];
}
