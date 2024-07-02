#ifndef preprocess_hpp
#define preprocess_hpp

#include "datatypes.hpp"
#include <vector>

extern Giro::MeshParams MP;
extern Giro::SolveParams SP;
extern std::vector<std::vector<float>> scadivmatrix, vecdivmatrix, scalapmatrix, veclapmatrix;
 
int preprocess();

#endif