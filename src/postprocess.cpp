#include <iostream>
#include <cstdarg>
#include "../dependencies/include/postprocess.hpp"
#include "../dependencies/include/preprocess.hpp"
#include "../dependencies/include/solve.hpp"

void postprocess(){
    std::cout << "This is the postprocess source" << std::endl;
}

// Recursive variadic template function
template<typename T, typename... Args>
void postprocess(T first, Args... args) {
    //Giro::Solve msv;
    //int ind = msv.matchscalartovar(first);
    writevth();
    postprocess(args...);
}

// Explicit instantiation to allow separate compilation
template void postprocess<>(const char* first);
template void postprocess<>(std::string first);
template void postprocess<>(const char* first, const char* second);
template void postprocess<>(std::string first, std::string second);
