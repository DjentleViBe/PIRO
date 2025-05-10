#ifndef bc_hpp
#define bc_hpp

#include <vector>
#include <CL/opencl.h>

extern std::vector<std::vector<int>> indices;
extern std::vector<int> indices_toprint;
extern std::vector<int> indices_toprint_vec;

namespace Piro::bc{
    void prepbc();
    void initbc();
    void readbc();
    void setbc();
    void opencl_setBC(int ind);
    void opencl_initBC();
    void opencl_setBC(int ind);
}

#endif