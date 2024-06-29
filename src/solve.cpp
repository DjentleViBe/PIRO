#include "../dependencies/include/solve.hpp"
#include "../dependencies/include/extras.hpp"
#include "../dependencies/include/operatoroverload.hpp"
#include <iostream>

int solve(){

    // apply boundary conditions

    // solve equations
    Giro::Solve solver;
    MP.AMR[0].CD[0].values = solver.ddt_r("T") - solver.ddc_r("Alpha", 1.0) * solver.laplacian_r("T");
    // update timestep
    return 0;
}