#include <stencils.hpp>
#include <mathoperations.hpp>
#include <datatypes.hpp>

void Piro::stencils::gradient_cd_x(Piro::CellData& cd, std::vector<uint> n, float norm){
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = Piro::math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                // Piro::logger::info(i);
                // index right side 
                if (x > 0) {
                    int right = Piro::math_operations::index(x - 1, y, z, n[0], n[1]); 
                    cd.columns.push_back(right);
                    cd.values.push_back(-1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
                }
                // index left side
                if (x < n[0] - 1) {
                    int left = Piro::math_operations::index(x + 1, y, z, n[0], n[1]);
                    cd.columns.push_back(left);
                    cd.values.push_back(1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
                }
            }
        }
    }
}

void Piro::stencils::gradient_cd_y(Piro::CellData& cd, std::vector<uint> n, float norm){
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = Piro::math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                // index right side 
                if (y > 0) {
                    int top = Piro::math_operations::index(x, y - 1, z, n[0], n[1]); 
                    cd.columns.push_back(top);
                    cd.values.push_back(-1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
                }
                // index left side
                if (y < n[1] - 1) {
                    int bottom = Piro::math_operations::index(x, y + 1, z, n[0], n[1]);
                    cd.columns.push_back(bottom);
                    cd.values.push_back(1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
            
                }
            }
        }
    }
}

void Piro::stencils::gradient_cd_z(Piro::CellData& cd, std::vector<uint> n, float norm){
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = Piro::math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                // Piro::logger::info(i);
                // index right side 
                if (z > 0) {
                    int up = Piro::math_operations::index(x, y, z - 1, n[0], n[1]); 
                    // Piro::logger::info(right);
                    cd.columns.push_back(up);
                    cd.values.push_back(-1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
                }
                // index left side
                if (z < n[2] - 1) {
                    int down = Piro::math_operations::index(x, y, z + 1, n[0], n[1]);
                    cd.columns.push_back(down);
                    cd.values.push_back(1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
            
                }
            }
        }
    }
}

void Piro::stencils::gradient_uw_x(Piro::CellData& cd, std::vector<uint> n, float norm){
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = Piro::math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                // Piro::logger::info(i);
                // index right side 
                if (x > 0) {
                    int right = Piro::math_operations::index(x - 1, y, z, n[0], n[1]); 
                    cd.columns.push_back(right);
                    cd.values.push_back(-1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
                }
                // index left side
                if (x < n[0] - 1) {
                    int left = Piro::math_operations::index(x, y, z, n[0], n[1]);
                    cd.columns.push_back(left);
                    cd.values.push_back(1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
            
                }
            }
        }
    }

}
void Piro::stencils::gradient_uw_y(Piro::CellData& cd, std::vector<uint> n, float norm){
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = Piro::math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                // index right side 
                if (y > 0) {
                    int top = Piro::math_operations::index(x, y - 1, z, n[0], n[1]); 
                    cd.columns.push_back(top);
                    cd.values.push_back(-1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
                }
                // index left side
                if (y < n[1] - 1) {
                    int bottom = Piro::math_operations::index(x, y, z, n[0], n[1]);
                    cd.columns.push_back(bottom);
                    cd.values.push_back(1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
            
                }
            }
        }
    }
}
void Piro::stencils::gradient_uw_z(Piro::CellData& cd, std::vector<uint> n, float norm){
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = Piro::math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                // Piro::logger::info(i);
                // index right side 
                if (z > 0) {
                    int up = Piro::math_operations::index(x, y, z - 1, n[0], n[1]); 
                    // Piro::logger::info(right);
                    cd.columns.push_back(up);
                    cd.values.push_back(-1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
                }
                // index left side
                if (z < n[2] - 1) {
                    int down = Piro::math_operations::index(x, y, z, n[0], n[1]);
                    cd.columns.push_back(down);
                    cd.values.push_back(1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
            
                }
            }
        }
    }
}

void Piro::stencils::gradient_dw_x(Piro::CellData& cd, std::vector<uint> n, float norm){
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = Piro::math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                // Piro::logger::info(i);
                // index right side 
                if (x > 0) {
                    int right = Piro::math_operations::index(x, y, z, n[0], n[1]); 
                    cd.columns.push_back(right);
                    cd.values.push_back(-1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
                }
                // index left side
                if (x < n[0] - 1) {
                    int left = Piro::math_operations::index(x + 1, y, z, n[0], n[1]);
                    cd.columns.push_back(left);
                    cd.values.push_back(1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
            
                }
            }
        }
    }
}
void Piro::stencils::gradient_dw_y(Piro::CellData& cd, std::vector<uint> n, float norm){
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = Piro::math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                // index right side 
                if (y > 0) {
                    int top = Piro::math_operations::index(x, y, z, n[0], n[1]); 
                    cd.columns.push_back(top);
                    cd.values.push_back(-1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
                }
                // index left side
                if (y < n[1] - 1) {
                    int bottom = Piro::math_operations::index(x, y + 1, z, n[0], n[1]);
                    cd.columns.push_back(bottom);
                    cd.values.push_back(1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
            
                }
            }
        }
    }
}
void Piro::stencils::gradient_dw_z(Piro::CellData& cd, std::vector<uint> n, float norm){
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = Piro::math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                // Piro::logger::info(i);
                // index right side 
                if (z > 0) {
                    int up = Piro::math_operations::index(x, y, z, n[0], n[1]); 
                    // Piro::logger::info(right);
                    cd.columns.push_back(up);
                    cd.values.push_back(-1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
                }
                // index left side
                if (z < n[2] - 1) {
                    int down = Piro::math_operations::index(x, y, z + 1, n[0], n[1]);
                    cd.columns.push_back(down);
                    cd.values.push_back(1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
            
                }
            }
        }
    }
}
        