#ifndef postprocess_hpp
#define postprocess_hpp

#include <datatypes.hpp>

namespace Piro::post{
    void export_paraview(float timestep);
}

namespace Piro::paraview{
        /**
         * @brief .vti export.
         *
         * This function generates a string in the .vti file format.
         *
         * @param AMR Cell data. 
         * @return string in .vti format.
         */
        std::string writevti(Piro::AMR AMR);
        /**
         * @brief .vth export.
         *
         * This function writes a .vth file for the current timestep.
         *
         * @param timestep Timestep number.
         */
        void writevth(int timestep);
}

#endif