#include <matrixgenerations.hpp>
#include <datatypes.hpp>
#include <openclutilities.hpp>
#include <CL/opencl.h>
#include <operatoroverload.hpp>
#include <logger.hpp>
#include <mathoperations.hpp>
#include <bc.hpp>
#include <preprocess.hpp>
#include <extras.hpp>
#include <printutilities.hpp>
#include <methods.hpp>
#include <operatorlaplacian.hpp>
#include <operatorgradient.hpp>
#include <string>

int Piro::matrix_generations::CSR::laplacian(){
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 3){
        Piro::CSR::laplacian_cd();
    }
    else{

    }
    
    return 0;
}

int Piro::matrix_generations::DENSE::laplacian(){
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 3){
        Piro::DENSE::laplacian_cd();
    }
    else{

    }
    
    return 0;
}

int Piro::matrix_generations::COO::laplacian(){
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 3){
         Piro::COO::laplacian_cd();
    }
    else{

    }
   
    return 0;
}

int Piro::matrix_generations::HT::laplacian(){
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 3){
        Piro::HT::laplacian_cd();
    }
    else{
        
    }
    return 0;
}

int Piro::matrix_generations::CSR::gradient(){
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 3){
        Piro::CSR::gradient_cd();
    }
    else{
        Piro::CSR::gradient_uw();
    }
    
    return 0;
}
