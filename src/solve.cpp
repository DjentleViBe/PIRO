#include "../dependencies/include/bc.hpp"
#include "../dependencies/include/solve.hpp"
#include "../dependencies/include/extras.hpp"
#include "../dependencies/include/operatoroverload.hpp"

#include <iostream>
#include <ctime>

char* dt;
std::time_t now;

int solve(){
    std::cout << "Solving . . ." << std::endl;
    // solve equations
    Giro::Solve solver;
    float time = 0.0;
    float totaltime = SP.totaltime;
    
    while(time < totaltime){
        print_time();
        Giro::scalarMatrix UEqn(solver.ddt_r("T") + solver.ddc_r("Alpha") * solver.laplacian_r("T"));
        print_time();
        //Giro::scalarMatrix UEqn(solver.ddt_r("Phi") + solver.grad_r("Phi", "U"));        
        UEqn.Solve(time);
        time += SP.timestep;
        if(DP.type != 0){
            break;
        }
    }
    
    return 0;
}