#ifndef kernelprocess_hpp
#define kernelprocess_hpp

#include <operatoroverload.hpp>
#include <datatypes.hpp>

namespace Piro::kernelmethods{
   namespace CSR{
      void TIMESCHEME_11(const std::vector<CLBuffer>& other, int N, int P, 
                     std::vector<uint> n,
                     float timestep, Piro::CLBuffer partC, 
                     Piro::CLBuffer partD, cl_mem RHS_vec, cl_mem LHS_vec);
   }
   namespace COO{
      void TIMESCHEME_11(const std::vector<CLBuffer>& other, int N, int nnz, 
                                        std::vector<uint> n, float timestep, 
                                        Piro::CLBuffer partC, Piro::CLBuffer partD, 
                                        cl_mem partB);
   }
   namespace DENSE{
      void TIMESCHEME_11(const std::vector<CLBuffer>& other, int N, int P,
                        float timestep,
                        cl_mem partB, Piro::CLBuffer partC,
                        Piro::CLBuffer partD);
   }
   namespace HT{
      void TIMESCHEME_11(const std::vector<CLBuffer>& other, int N, 
                        int TABLE_SIZE, cl_mem partB, Piro::CLBuffer partC,
                        Piro::CLBuffer partD, float timestep);
   }
}
#endif