#ifndef bc_hpp
#define bc_hpp

#include <vector>
#include <CL/opencl.h>

extern std::vector<std::vector<int>> indices;
extern std::vector<int> indices_toprint;
extern std::vector<int> indices_toprint_vec;
extern cl_int err;
void readbc();
void setbc();
void opencl_setBC(int ind);

#endif