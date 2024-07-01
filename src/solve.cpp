#include "../dependencies/include/bc.hpp"
#include "../dependencies/include/solve.hpp"
#include "../dependencies/include/extras.hpp"
#include "../dependencies/include/operatoroverload.hpp"
#include <iostream>

int solve(){
    std::cout << "Solving . . ." << std::endl;
    // solve equations
    Giro::Solve solver;
    MP.AMR[0].CD[0].values = solver.ddt_r("T") - solver.ddc_r("Alpha", 1.0) * solver.laplacian_r("T");
    // apply boundary conditions
    setbc();
    // update timestep
    return 0;
}