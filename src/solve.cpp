#include "../dependencies/include/bc.hpp"
#include "../dependencies/include/solve.hpp"
#include "../dependencies/include/extras.hpp"
#include "../dependencies/include/operatoroverload.hpp"
#include <iostream>

int solve(){
    std::cout << "Solving . . ." << std::endl;
    // solve equations
    Giro::Solve solver;
    float time = 0.0;
    while(time < SP.totaltime){
        //Giro::scalarMatrix UEqn(solver.ddt_r("T") + solver.ddc_r("Alpha") * solver.laplacian_r("T"));
        Giro::scalarMatrix UEqn(solver.ddt_r("Phi") + solver.grad_r("Phi", "U"));        
        UEqn.Solve(time);
        time += SP.timestep;
    }
    
    return 0;
}