// poisson_1d.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>

const double pi = 3.14159265358979323846;

double f(double x) {
    return pi * pi * sin(pi * x);
}

// Analytical solution for comparison: u(x) = (1 - x) * x / 2
double u_exact(double x) {
    return sin(pi * x);
}

int main() {
    int n_elements = 10;
    int n_nodes = n_elements + 1;
    double L = 1.0;
    double h = L / n_elements;

    std::vector<double> nodes(n_nodes);
    for (int i = 0; i < n_nodes; ++i)
        nodes[i] = i * h;

    std::vector<std::vector<double>> K(n_nodes, std::vector<double>(n_nodes, 0.0));
    std::vector<double> F(n_nodes, 0.0);

    // Assembly loop
    for (int e = 0; e < n_elements; ++e) {
        int i = e;
        int j = e + 1;

        // Local stiffness matrix
        double ke[2][2] = {{1.0, -1.0}, {-1.0, 1.0}};
        // Local load vector
        double fe[2] = {
            f(nodes[i]) * h / 2.0,
            f(nodes[j]) * h / 2.0
        };

        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b)
                K[e + a][e + b] += ke[a][b] / h;
            F[e + a] += fe[a];
        }
    }

    // Apply Dirichlet BCs: u(0) = u(1) = 0
    K[0][0] = 1.0;
    K[0][1] = 0.0;
    F[0] = 0.0;

    K[n_nodes - 1][n_nodes - 1] = 1.0;
    K[n_nodes - 1][n_nodes - 2] = 0.0;
    F[n_nodes - 1] = 0.0;

    // Solve linear system using naive Gauss elimination (for small systems)
    std::vector<double> u(n_nodes, 0.0);
    std::vector<std::vector<double>> A = K;
    std::vector<double> b = F;

    for (int i = 0; i < n_nodes; ++i) {
        double pivot = A[i][i];
        for (int j = i; j < n_nodes; ++j)
            A[i][j] /= pivot;
        b[i] /= pivot;

        for (int k = i + 1; k < n_nodes; ++k) {
            double factor = A[k][i];
            for (int j = i; j < n_nodes; ++j)
                A[k][j] -= factor * A[i][j];
            b[k] -= factor * b[i];
        }
    }

    for (int i = n_nodes - 1; i >= 0; --i) {
        u[i] = b[i];
        for (int j = i + 1; j < n_nodes; ++j)
            u[i] -= A[i][j] * u[j];
    }

    // Output results to file
    std::ofstream file("solution.csv");
    file << "x,approx,exact\n";
    for (int i = 0; i < n_nodes; ++i)
        file << nodes[i] << "," << u[i] << "," << u_exact(nodes[i]) << "\n";
    file.close();

    std::cout << "Solution written to solution.csv. You can plot it using Python or another tool." << std::endl;
    return 0;
}
