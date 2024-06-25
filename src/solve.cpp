#include "../dependencies/include/solve.hpp"
#include "../dependencies/include/extras.hpp"
#include "../dependencies/include/operatoroverload.hpp"
#include <iostream>

int solve(){

    // apply boundary conditions

    // solve equations
    Giro::Solve solver;
    //printMatrix(solver.ddt("T"));
    //printMatrix(solver.ddc("Alpha", 5.0));
    printMatrix(solver.laplacian("T"));
    printMatrix(solver.ddt("T") + solver.ddc("Alpha", 5.0));
    // update timestep
    return 0;
}