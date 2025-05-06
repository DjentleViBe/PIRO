#include <bc.hpp>
#include <process.hpp>
#include <extras.hpp>
#include <operatoroverload.hpp>
#include <printutilities.hpp>
#include <iostream>
#include <ctime>
#include <logger.hpp>

char* dt;
std::time_t now;

int solve(){
    Piro::logger::info("Solving . . .\n");
    // solve equations
    Piro::process solver;
    float time = 0.0;
    float totaltime = SP.totaltime;
    while(time < totaltime){
        // Heat Equation
        // Piro::scalarMatrix UEqn(solver.ddt_r("U") + (solver.ddc_r("Alpha") * solver.laplacian_full("U")));
        // Advection equation 
        // Piro::scalarMatrix UEqn(solver.ddt_r("T") - solver.div_r("T", "U"));        
        // Piro::scalarMatrix UEqn(solver.ddt_r("U") + solver.div_r("U", "U") - solver.laplacian_r("U"));        
        // Piro::scalarMatrix UEqn(solver.ddc_r("Hbar") * solver.ddt_r("Psi") + \
        //                        solver.ddc_r("Hbar2m") * solver.laplacian_r("Psi") - \
        //                        solver.r("Psi"));    
        Piro::scalarMatrix UEqn(solver.ddt_r("U") = solver.laplacian_CSR("U", "Alpha")); 
        //Piro::scalarMatrix UEqn(solver.ddt_r("U") = solver.laplacian_CSR("U")); 
        UEqn.Solve(time);
        time += SP.timestep;
        
        //if(DP.type != 0){
        //    break;
        //}
    }
    return 0;
}