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
        int si = SP.getvalue<int>(Piro::SolveParams::SIMINDEX);
        switch(si){
            case 0: {
                // Heat Eq : 
                Piro::scalarMatrix UEqn(solver.ddt_r("T") = solver.laplacian("T", "Alpha"));
                UEqn.Solve(time);
                break;
            }
            case 1: {
                // Advection Eq :
                Piro::scalarMatrix UEqn(solver.ddt_r("U") = solver.vector("v") & solver.gradient("U"));
                UEqn.Solve(time);
                break;
            }
            case 2: {
                // Continuity Eq : 
                Piro::scalarMatrix UEqn(solver.ddt_r("rho") = solver.div("rho", "u"));
                UEqn.Solve(time);
                break;
            }    
            default: {
                Piro::logger::info("Invalid sim index");
                std::exit(1);
            }
        }
        time += SP.getvalue<float>(Piro::SolveParams::TIMESTEP);
    }
    return 0;
}