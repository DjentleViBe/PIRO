#include "../dependencies/include/solve.hpp"
#include "../dependencies/include/extras.hpp"
#include <iostream>

int solve(){

    // apply boundary conditions

    // solve equations
    Giro::Solve solver;
    printMatrix(solver.ddt("P"));
    // update timestep
    std::cout << "solve" << std::endl;
    return 0;
}