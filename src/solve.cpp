#include "../dependencies/include/solve.hpp"
#include "../dependencies/include/extras.hpp"
#include <iostream>

int solve(){

    // apply boundary conditions

    // solve equations
    Giro::Solve solver;
    printMatrix(solver.ddt("P"));
    printMatrix(solver.ddc("Alpha", 5.0));
    printMatrix(solver.laplacian("V"));
    // update timestep
    return 0;
}