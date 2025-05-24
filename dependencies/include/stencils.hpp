#ifndef stencils_hpp
#define stencils_hpp

#include <datatypes.hpp>

namespace Piro{
    namespace CSR{
        namespace stencils{
            void gradient_uw_x(Piro::CellData& cd, std::vector<uint> n, float norm);
            void gradient_uw_y(Piro::CellData& cd, std::vector<uint> n, float norm);
            void gradient_uw_z(Piro::CellData& cd, std::vector<uint> n, float norm);
            void gradient_dw_x(Piro::CellData& cd, std::vector<uint> n, float norm);
            void gradient_dw_y(Piro::CellData& cd, std::vector<uint> n, float norm);
            void gradient_dw_z(Piro::CellData& cd, std::vector<uint> n, float norm);
            void gradient_cd_x(Piro::CellData& cd, std::vector<uint> n, float norm);
            void gradient_cd_y(Piro::CellData& cd, std::vector<uint> n, float norm);
            void gradient_cd_z(Piro::CellData& cd, std::vector<uint> n, float norm);

            void laplacian_cd(Piro::CellData& cd, std::vector<uint> n, float norm);
            void laplacian_uw(Piro::CellData& cd, std::vector<uint> n, float norm);
            void laplacian_dw(Piro::CellData& cd, std::vector<uint> n, float norm);

            void div_uw_x(Piro::CellData& cd, std::vector<uint> n, float norm);
            void div_uw_y(Piro::CellData& cd, std::vector<uint> n, float norm);
            void div_uw_z(Piro::CellData& cd, std::vector<uint> n, float norm);
            void div_cd_x(Piro::CellData& cd, std::vector<uint> n, float norm);
            void div_cd_y(Piro::CellData& cd, std::vector<uint> n, float norm);
            void div_cd_z(Piro::CellData& cd, std::vector<uint> n, float norm);
            void div_dw_x(Piro::CellData& cd, std::vector<uint> n, float norm);
            void div_dw_y(Piro::CellData& cd, std::vector<uint> n, float norm);
            void div_dw_z(Piro::CellData& cd, std::vector<uint> n, float norm);

        }
    }
    namespace DENSE{
        namespace stencils{
            void laplacian_cd(Piro::CellData& cd, std::vector<uint> n, float norm);
            void laplacian_uw(Piro::CellData& cd, std::vector<uint> n, float norm);
            void laplacian_dw(Piro::CellData& cd, std::vector<uint> n, float norm);
        }
    }

    namespace COO{
        namespace stencils{
            void laplacian_cd(Piro::CellData& cd, std::vector<uint> n, float norm);
            void laplacian_uw(Piro::CellData& cd, std::vector<uint> n, float norm);
            void laplacian_dw(Piro::CellData& cd, std::vector<uint> n, float norm);
        }
    }

     namespace HT{
        namespace stencils{
            void laplacian_cd(Piro::CellData& cd, std::vector<uint> n, float norm);
            void laplacian_uw(Piro::CellData& cd, std::vector<uint> n, float norm);
            void laplacian_dw(Piro::CellData& cd, std::vector<uint> n, float norm);
        }
    }
}

#endif