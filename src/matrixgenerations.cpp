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
#include <operatordivergence.hpp>
#include <operatordivergence.hpp>

#include <string>

int Piro::matrix_generations::CSR::laplacian(){
    Piro::CSR::laplacian();
    return 0;
}

int Piro::matrix_generations::DENSE::laplacian(){
    Piro::DENSE::laplacian();
    return 0;
}

int Piro::matrix_generations::COO::laplacian(){
    Piro::COO::laplacian();
    return 0;
}

int Piro::matrix_generations::HT::laplacian(){
    Piro::HT::laplacian();
    return 0;
}

int Piro::matrix_generations::CSR::gradient(){
    Piro::CSR::gradient();
    return 0;
}

int Piro::matrix_generations::CSR::div(){
    Piro::CSR::div();
    return 0;
}

int Piro::matrix_generations::COO::div(){
    Piro::CSR::div();
    return 0;
}

int Piro::matrix_generations::DENSE::div(){
    Piro::CSR::div();
    return 0;
}

int Piro::matrix_generations::HT::div(){
    Piro::CSR::div();
    return 0;
}
