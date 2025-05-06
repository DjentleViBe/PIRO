#ifndef postprocess_hpp
#define postprocess_hpp

#include <datatypes.hpp>

namespace Piro::post{
    void export_paraview(float timestep);
}

namespace Piro::paraview{
        std::string writevti(Piro::AMR AMR);
        void writevth(int timestep);
}

#endif