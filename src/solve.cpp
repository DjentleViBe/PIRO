#include <bc.hpp>
#include <process.hpp>
#include <extras.hpp>
#include <operatoroverload.hpp>
#include <printutilities.hpp>
#include <iostream>
#include <ctime>
#include <logger.hpp>
#include <datatypes.hpp>

int Piro::solve(){
    Piro::logger::info("Solving . . .\n");
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    float totaltime = SP.getvalue<float>(Piro::SolveParams::TOTALTIME);
    // solve equations
    Piro::process solver;
    float time = 0.0;
    while(time < totaltime){
        // Heat Equation
        // Piro::scalarMatrix UEqn(solver.ddt_r("U") + (solver.ddc_r("Alpha") * solver.laplacian_full("U")));
        // Advection equation 
        // Piro::scalarMatrix UEqn(solver.ddt_r("T") - solver.div_r("T", "U"));        
        // Piro::scalarMatrix UEqn(solver.ddt_r("U") + solver.div_r("U", "U") - solver.laplacian_r("U"));        
        // Piro::scalarMatrix UEqn(solver.ddc_r("Hbar") * solver.ddt_r("Psi") + \
        //                        solver.ddc_r("Hbar2m") * solver.laplacian_r("Psi") - \
        //                        solver.r("Psi"));    
        Piro::scalarMatrix UEqn(solver.ddt_r("U") = solver.laplacian("U", "Alpha"));
        UEqn.Solve(time);
        time += SP.getvalue<float>(Piro::SolveParams::TIMESTEP);
        
        //if(DP.type != 0){
        //    break;
        //}
    }
    return 0;
}