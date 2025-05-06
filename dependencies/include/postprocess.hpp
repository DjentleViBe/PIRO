#ifndef postprocess_hpp
#define postprocess_hpp
#include <datatypes.hpp>
//void postprocess();

// template<typename T, typename... Args>
// void postprocess(T first, Args... args);
namespace Piro::post{
    void export_paraview(float timestep);
}

namespace Piro::paraview{
        std::string writevti(Piro::AMR AMR);
        void writevth(int timestep);
}

#endif