#ifndef preprocess_hpp
#define preprocess_hpp

#include "datatypes.hpp"
#include <vector>

extern Giro::MeshParams MP;
extern Giro::SolveParams SP;
extern std::vector<std::vector<float>> scagradmatrix, scadivmatrix;
extern std::vector<float> scalapvector;
extern int ts;

int preprocess();
int idx(int i, int j, int k, int N_x, int N_y);

#endif