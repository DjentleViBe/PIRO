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
        // Piro::scalarMatrix UEqn(solver.ddt_r("U") + solver.div_r("U", "U") - solver.laplacian_r("U"));

        // Heat Eq : 
        // Piro::scalarMatrix UEqn(solver.ddt_r("U") = solver.laplacian("U", "Alpha"));
        // Advection Eq :
        // Piro::scalarMatrix UEqn(solver.ddt_r("U") = solver.vector("v") & solver.gradient("U"));
        // Continuity Eq : 
        Piro::scalarMatrix UEqn(solver.ddt_r("rho") = solver.div("rho", "u"));
        UEqn.Solve(time);
        time += SP.getvalue<float>(Piro::SolveParams::TIMESTEP);
        
        //if(DP.type != 0){
        //    break;
        //}
    }
    return 0;
}