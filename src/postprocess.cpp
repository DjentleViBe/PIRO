#include <iostream>
#include <cstdarg>
#include <postprocess.hpp>
#include <preprocess.hpp>
#include <solve.hpp>

void postprocess(){
    // std::cout << "This is the postprocess source" << std::endl;
}

// Recursive variadic template function
template<typename T, typename... Args>
void postprocess(T first, Args... args) {
    //Piro::Solve msv;
    //int ind = msv.matchscalartovar(first);
    writevth(ts);
    postprocess(args...);
    std::cout << "File exported" << std::endl;
}

// Explicit instantiation to allow separate compilation
template void postprocess<>(const char* first);
template void postprocess<>(std::string first);
template void postprocess<>(const char* first, const char* second);
template void postprocess<>(std::string first, std::string second);
