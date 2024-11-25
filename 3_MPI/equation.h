#ifndef EQUATION_H
#define EQUATION_H
#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

#define TIME_STEPS 20
typedef std::vector< std::vector<double>> VVEC;
typedef std::vector<double> VDOUB;
typedef std::vector<int> VINT;

class Grid {
  public:
    int N;
    double Lx, Ly, Lz, h_x, h_y, h_z, tau;
    char* L_type;

    Grid() {}

    Grid(int N, char* L_type, double L) {
        this->N = N;
        this->L_type = L_type;
        this->Lx = this->Ly = this->Lz = L;
        this->h_x = this->h_y = this->h_z = L / N;
        this->tau = 0.0001;
    }

    Grid(int N, char* L_type, double Lx, double Ly, double Lz) {
        this->N = N;
        this->L_type = L_type;
        this->Lx = Lx;
        this->Ly = Ly;
        this->Lz = Lz;
        this->h_x = Lx / N;
        this->h_y = Ly / N;
        this->h_z = Lz / N;
        this->tau = 0.0001;
    }

    inline int index(const int& i, const int& j, const int& k) const {
        return (i * (this->N + 1) + j) * (this->N + 1) + k;
    }
};

class Block {
  public:
    int N, Nx, Ny, Nz, padded_Nx, padded_Ny, padded_Nz, x_start, y_start, z_start, x_end, y_end, z_end, rank, dim0_rank, dim1_rank, dim2_rank;
    VINT neighbors;
    VDOUB left_send, right_send, bottom_send, top_send, front_send, back_send;
    VDOUB left_recieve, right_recieve, bottom_recieve, top_recieve, front_recieve, back_recieve;

    Block(const Grid& g, VINT& neighbors, int coords[], const int& dim0_n, const int& dim1_n, const int& dim2_n, int& rank) {
        this->rank = rank;
        this->dim0_rank = coords[0];
        this->dim1_rank = coords[1];
        this->dim2_rank = coords[2];

        int base_size_x = (g.N + 1) / dim1_n;
        int remainder_x = (g.N + 1) % dim1_n;
        int base_size_y = (g.N + 1) / dim0_n;
        int remainder_y = (g.N + 1) % dim0_n;
        int base_size_z = (g.N + 1) / dim2_n;
        int remainder_z = (g.N + 1) % dim2_n;

        // local block sizes by each dim
        this->Nx = (this->dim1_rank < dim1_n - remainder_x) ? base_size_x : (base_size_x + 1);
        this->Ny = (this->dim0_rank < remainder_y) ? (base_size_y + 1) : base_size_y;
        this->Nz = (this->dim2_rank < remainder_z) ? (base_size_z + 1) : base_size_z;

        // make fixed length vectors for data transmission between processes
        this->left_send.resize(this->Nx * this->Nz);
        this->left_recieve.resize(this->Nx * this->Nz);
        this->right_send.resize(this->Nx * this->Nz);
        this->right_recieve.resize(this->Nx * this->Nz);
        this->bottom_send.resize(this->Ny * this->Nz);
        this->bottom_recieve.resize(this->Ny * this->Nz);
        this->top_send.resize(this->Ny * this->Nz);
        this->top_recieve.resize(this->Ny * this->Nz);
        this->front_send.resize(this->Nx * this->Ny);
        this->front_recieve.resize(this->Nx * this->Ny);
        this->back_send.resize(this->Nx * this->Ny);
        this->back_recieve.resize(this->Nx * this->Ny);

        // global grid start and end indices
        this->x_end = g.N;
        this->y_start = 0;
        this->z_start = 0;

        for (int i = dim1_rank; i > 0; --i)
            this->x_end -= (i <= dim1_n - remainder_x) ? base_size_x : (base_size_x + 1);
        this->x_start = this->x_end - this->Nx + 1;

        for (int i = 0; i < dim0_rank; ++i)
            this->y_start += (i < remainder_y) ? (base_size_y + 1) : base_size_y;
        this->y_end = this->y_start + this->Ny - 1;

        for (int i = 0; i < dim2_rank; ++i)
            this->z_start += (i < remainder_z) ? (base_size_z + 1) : base_size_z;
        this->z_end = this->z_start + this->Nz - 1;

        // +2 for ghost layers (halo)
        this->padded_Nx = this->Nx + 2;
        this->padded_Ny = this->Ny + 2;
        this->padded_Nz = this->Nz + 2;

        this->N = this->padded_Nx * this->padded_Ny * this->padded_Nz;
        this->neighbors = neighbors;
    }

    inline int index(const int& i, const int& j, const int& k) const {
        return i * (this->padded_Ny * this->padded_Nz) + j * this->padded_Nz + k;
    }

    void print_block_info() const {
        std::cout << "Block " << this->rank << " cooord ("<< this->dim0_rank << "," << this->dim1_rank << "," << this->dim2_rank << ") info:\n\t" <<
            "Nx = " << this->Nx << "\n\t\t" << "x_start = " << this->x_start << "\n\t\t" << "x_end = " << this->x_end << "\n\t" <<
                "Ny = " << this->Ny << "\n\t\t" << "y_start = " << this->y_start << "\n\t\t" << "y_end = " << this->y_end << "\n\t" <<
                    "Nz = " << this->Nz << "\n\t\t" << "z_start = " << this->z_start << "\n\t\t" << "z_end = " << this->z_end << "\n\t" <<  std::endl;
    }
};

// u_analytical variant 8
inline double u_analytical(const Grid& g, const double& x, const double& y,const double& z, const double& t) {
    return sin((2 * M_PI * x) / g.Lx) * sin((4 * M_PI * y) / g.Ly) * sin((6 * M_PI * z) / g.Lz) *
           cos(M_PI * sqrt((4 / pow(g.Lx, 2)) + (16 / pow(g.Ly, 2)) + (36 / pow(g.Lz, 2))) * t);
}

void solve_equation(const Grid& grid, Block& block, const int& dim0_n, const int& dim1_n, const int& dim2_n, MPI_Comm& comm_cart, double& time, double& max_inaccuracy, double& first_step_inaccuracy, double& last_step_inaccuracy, VDOUB& result);

#endif //EQUATION_H
