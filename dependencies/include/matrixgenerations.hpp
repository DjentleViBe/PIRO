#ifndef mathgeneration_hpp
#define mathgeneration_hpp

#include <string>

namespace Piro::matrix_generations{
    namespace CSR{  
        /**
         * @brief Generation of the laplacian matrix.
         *
         * This function generates the laplacian matrix.
         *
         * @param n number to be tested. 
         * @return TRUE if successful.
         */
        int laplacian();
        int gradient();
        int div();
    }
    namespace DENSE{
        int laplacian();
        int div();
    }
    namespace HT{
        int laplacian();
        int div();
    }
    namespace COO{
        int laplacian();
        int div();
    }
}


#endif