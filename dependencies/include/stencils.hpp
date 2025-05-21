#ifndef stencils_hpp
#define stencils_hpp

#include <datatypes.hpp>

namespace Piro{
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

    }
}

#endif