#ifndef EQUATION_H
#define EQUATION_H
#include <iostream>
#include <vector>
#include <cmath>

#define TIME_STEPS 20
typedef std::vector< std::vector<double>> VVEC;
typedef std::vector<double> VDOUB;

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

// u_analytical variant 8
inline double u_analytical(const Grid& g, const double& x, const double& y,const double& z, const double& t) {
    return sin((2 * M_PI * x) / g.Lx) * sin((4 * M_PI * y) / g.Ly) * sin((6 * M_PI * z) / g.Lz) *
           cos(M_PI * sqrt((4 / pow(g.Lx, 2)) + (16 / pow(g.Ly, 2)) + (36 / pow(g.Lz, 2))) * t);
}

void solve_equation(Grid& grid, double& time, double& max_inaccuracy, double& first_step_inaccuracy, double& last_step_inaccuracy, VDOUB& result, int& threads_num);

#endif //EQUATION_H
