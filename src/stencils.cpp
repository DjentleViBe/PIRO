#include <stencils.hpp>
#include <mathoperations.hpp>
#include <datatypes.hpp>
#include <bc.hpp>

void Piro::CSR::stencils::gradient_cd_x(Piro::CellData& cd, std::vector<uint> n, float norm){
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
void Piro::CSR::stencils::gradient_cd_y(Piro::CellData& cd, std::vector<uint> n, float norm){
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
void Piro::CSR::stencils::gradient_cd_z(Piro::CellData& cd, std::vector<uint> n, float norm){
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
void Piro::CSR::stencils::gradient_uw_x(Piro::CellData& cd, std::vector<uint> n, float norm){
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
void Piro::CSR::stencils::gradient_uw_y(Piro::CellData& cd, std::vector<uint> n, float norm){
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
void Piro::CSR::stencils::gradient_uw_z(Piro::CellData& cd, std::vector<uint> n, float norm){
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
void Piro::CSR::stencils::gradient_dw_x(Piro::CellData& cd, std::vector<uint> n, float norm){
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
void Piro::CSR::stencils::gradient_dw_y(Piro::CellData& cd, std::vector<uint> n, float norm){
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
void Piro::CSR::stencils::gradient_dw_z(Piro::CellData& cd, std::vector<uint> n, float norm){
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

void Piro::CSR::stencils::laplacian_cd(Piro::CellData& cd, std::vector<uint> n, float norm){
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                
                // Self connection (central point)
                cd.columns.push_back(i);
                cd.values.push_back(-6.0/norm);
                
                // Neighbors in the x-direction (x±1)
                if (x > 0) {
                    cd.columns.push_back(math_operations::index(x-1, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }
                if (x < n[0] - 1) {
                    cd.columns.push_back(math_operations::index(x+1, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                
                }

                // Neighbors in the y-direction (y±1)
                if (y > 0) {
                    cd.columns.push_back(math_operations::index(x, y-1, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }
                if (y < n[1] - 1) {
                    cd.columns.push_back(math_operations::index(x, y+1, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }

                // Neighbors in the z-direction (z±1)
                if (z > 0) {
                    cd.columns.push_back(math_operations::index(x, y, z-1, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }
                if (z < n[2] - 1) {
                    cd.columns.push_back(math_operations::index(x, y, z+1, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                }
            
                if (Piro::bc::isBoundaryPoint(x, y, z)) {
                    auto it = std::find(cd.columns.begin(),
                    cd.columns.end(), i);
    
                    if (it == cd.columns.end()) {
                        // Add the boundary point if not already in the matrix.
                        cd.columns.push_back(i); // for diagonal entry
                        cd.values.push_back(10.0); // large value to enforce the BC
                    }
                    else{
                        size_t index = std::distance(cd.columns.begin(), it);
                        cd.values[index] = 10.0;
                    }
                }

                cd.rowpointers[i + 1] = cd.columns.size();
            }
        }
    }
}

void Piro::CSR::stencils::laplacian_uw(Piro::CellData& cd, std::vector<uint> n, float norm){
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                
                // Self connection (central point)
                cd.columns.push_back(i);
                cd.values.push_back(-6.0/norm);
                
                // Neighbors in the x-direction (x±1)
                if (x > 0) {
                    cd.columns.push_back(math_operations::index(x-1, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }
                if (x < n[0] - 1) {
                    cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                
                }

                // Neighbors in the y-direction (y±1)
                if (y > 0) {
                    cd.columns.push_back(math_operations::index(x, y-1, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }
                if (y < n[1] - 1) {
                    cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }

                // Neighbors in the z-direction (z±1)
                if (z > 0) {
                    cd.columns.push_back(math_operations::index(x, y, z-1, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }
                if (z < n[2] - 1) {
                    cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                }
            
                if (Piro::bc::isBoundaryPoint(x, y, z)) {
                    auto it = std::find(cd.columns.begin(),
                    cd.columns.end(), i);
    
                    if (it == cd.columns.end()) {
                        // Add the boundary point if not already in the matrix.
                        cd.columns.push_back(i); // for diagonal entry
                        cd.values.push_back(10.0); // large value to enforce the BC
                    }
                    else{
                        size_t index = std::distance(cd.columns.begin(), it);
                        cd.values[index] = 10.0;
                    }
                }

                cd.rowpointers[i + 1] = cd.columns.size();
            }
        }
    }
}

void Piro::CSR::stencils::laplacian_dw(Piro::CellData& cd, std::vector<uint> n, float norm){
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                
                // Self connection (central point)
                cd.columns.push_back(i);
                cd.values.push_back(-6.0/norm);
                
                // Neighbors in the x-direction (x±1)
                if (x > 0) {
                    cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }
                if (x < n[0] - 1) {
                    cd.columns.push_back(math_operations::index(x + 1, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                
                }

                // Neighbors in the y-direction (y±1)
                if (y > 0) {
                    cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }
                if (y < n[1] - 1) {
                    cd.columns.push_back(math_operations::index(x, y + 1, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }

                // Neighbors in the z-direction (z±1)
                if (z > 0) {
                    cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }
                if (z < n[2] - 1) {
                    cd.columns.push_back(math_operations::index(x, y, z + 1, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                }
            
                if (Piro::bc::isBoundaryPoint(x, y, z)) {
                    auto it = std::find(cd.columns.begin(),
                    cd.columns.end(), i);
    
                    if (it == cd.columns.end()) {
                        // Add the boundary point if not already in the matrix.
                        cd.columns.push_back(i); // for diagonal entry
                        cd.values.push_back(10.0); // large value to enforce the BC
                    }
                    else{
                        size_t index = std::distance(cd.columns.begin(), it);
                        cd.values[index] = 10.0;
                    }
                }

                cd.rowpointers[i + 1] = cd.columns.size();
            }
        }
    }
}

void Piro::DENSE::stencils::laplacian_cd(Piro::CellData& cd, std::vector<uint> n, float norm){
    size_t total_size = n[0] * n[1] * n[2];
    for (int k = 0; k < n[2]; ++k) {
        for (int j = 0; j < n[1]; ++j) {
            for (int i = 0; i < n[0]; ++i) {
                int l = Piro::math_operations::idx(i, j, k, n[0], n[1]);

                // Diagonal entry
                cd.matrix[l * total_size + l] = -6 / norm;
                //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l * total_size + l] = 0;

                // Off-diagonal entries
                if (i > 0) {
                    cd.matrix[l * total_size + Piro::math_operations::idx(i-1, j, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l* total_size + Piro::math_operations::idx(i-1, j, k, n[0], n[1])] = -timestep / (2 * delta[0]);
                    
                }
                if (i < n[0] - 1) {
                    cd.matrix[l * total_size + Piro::math_operations::idx(i+1, j, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l * total_size + Piro::math_operations::idx(i+1, j, k, n[0], n[1])] = timestep / (2 * delta[0]);
                    
                }
                if (j > 0) {
                    cd.matrix[l * total_size + Piro::math_operations::idx(i, j-1, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l * total_size + Piro::math_operations::idx(i, j-1, k, n[0], n[1])] = -timestep / (2 * delta[1]);
                    
                }
                if (j < n[1] - 1) {
                    cd.matrix[l * total_size + Piro::math_operations::idx(i, j+1, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l * total_size + Piro::math_operations::idx(i, j+1, k, n[0], n[1])] = timestep / (2 * delta[1]);
                    
                }
                if (k > 0) {
                    cd.matrix[l* total_size + Piro::math_operations::idx(i, j, k-1, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l* total_size + Piro::math_operations::idx(i, j, k-1, n[0], n[1])] = -timestep / (2 * delta[2]);
                    
                }
                if (k < n[2] - 1) {
                    cd.matrix[l* total_size + Piro::math_operations::idx(i, j, k+1, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l* total_size + Piro::math_operations::idx(i, j, k+1, n[0], n[1])] = timestep / (2 * delta[2]);
                    
                }
            }
        }
    }
}

void Piro::DENSE::stencils::laplacian_uw(Piro::CellData& cd, std::vector<uint> n, float norm){
    size_t total_size = n[0] * n[1] * n[2];
    for (int k = 0; k < n[2]; ++k) {
        for (int j = 0; j < n[1]; ++j) {
            for (int i = 0; i < n[0]; ++i) {
                int l = Piro::math_operations::idx(i, j, k, n[0], n[1]);

                // Diagonal entry
                cd.matrix[l * total_size + l] = -6 / norm;
                //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l * total_size + l] = 0;

                // Off-diagonal entries
                if (i > 0) {
                    cd.matrix[l * total_size + Piro::math_operations::idx(i-1, j, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l* total_size + Piro::math_operations::idx(i-1, j, k, n[0], n[1])] = -timestep / (2 * delta[0]);
                    
                }
                if (i < n[0] - 1) {
                    cd.matrix[l * total_size + Piro::math_operations::idx(i, j, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l * total_size + Piro::math_operations::idx(i+1, j, k, n[0], n[1])] = timestep / (2 * delta[0]);
                    
                }
                if (j > 0) {
                    cd.matrix[l * total_size + Piro::math_operations::idx(i, j-1, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l * total_size + Piro::math_operations::idx(i, j-1, k, n[0], n[1])] = -timestep / (2 * delta[1]);
                    
                }
                if (j < n[1] - 1) {
                    cd.matrix[l * total_size + Piro::math_operations::idx(i, j, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l * total_size + Piro::math_operations::idx(i, j+1, k, n[0], n[1])] = timestep / (2 * delta[1]);
                    
                }
                if (k > 0) {
                    cd.matrix[l* total_size + Piro::math_operations::idx(i, j, k-1, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l* total_size + Piro::math_operations::idx(i, j, k-1, n[0], n[1])] = -timestep / (2 * delta[2]);
                    
                }
                if (k < n[2] - 1) {
                    cd.matrix[l* total_size + Piro::math_operations::idx(i, j, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l* total_size + Piro::math_operations::idx(i, j, k+1, n[0], n[1])] = timestep / (2 * delta[2]);
                    
                }
            }
        }
    }
}

void Piro::DENSE::stencils::laplacian_dw(Piro::CellData& cd, std::vector<uint> n, float norm){
    size_t total_size = n[0] * n[1] * n[2];
    for (int k = 0; k < n[2]; ++k) {
        for (int j = 0; j < n[1]; ++j) {
            for (int i = 0; i < n[0]; ++i) {
                int l = Piro::math_operations::idx(i, j, k, n[0], n[1]);

                // Diagonal entry
                cd.matrix[l * total_size + l] = -6 / norm;
                //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l * total_size + l] = 0;

                // Off-diagonal entries
                if (i > 0) {
                    cd.matrix[l * total_size + Piro::math_operations::idx(i, j, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l* total_size + Piro::math_operations::idx(i-1, j, k, n[0], n[1])] = -timestep / (2 * delta[0]);
                    
                }
                if (i < n[0] - 1) {
                    cd.matrix[l * total_size + Piro::math_operations::idx(i + 1, j, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l * total_size + Piro::math_operations::idx(i+1, j, k, n[0], n[1])] = timestep / (2 * delta[0]);
                    
                }
                if (j > 0) {
                    cd.matrix[l * total_size + Piro::math_operations::idx(i, j, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l * total_size + Piro::math_operations::idx(i, j-1, k, n[0], n[1])] = -timestep / (2 * delta[1]);
                    
                }
                if (j < n[1] - 1) {
                    cd.matrix[l * total_size + Piro::math_operations::idx(i, j + 1, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l * total_size + Piro::math_operations::idx(i, j+1, k, n[0], n[1])] = timestep / (2 * delta[1]);
                    
                }
                if (k > 0) {
                    cd.matrix[l* total_size + Piro::math_operations::idx(i, j, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l* total_size + Piro::math_operations::idx(i, j, k-1, n[0], n[1])] = -timestep / (2 * delta[2]);
                    
                }
                if (k < n[2] - 1) {
                    cd.matrix[l* total_size + Piro::math_operations::idx(i, j, k + 1, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l* total_size + Piro::math_operations::idx(i, j, k+1, n[0], n[1])] = timestep / (2 * delta[2]);
                    
                }
            }
        }
    }
}

void Piro::COO::stencils::laplacian_cd(Piro::CellData& cd, std::vector<uint> n, float norm){
    int N = n[0] * n[1] * n[2];
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = Piro::math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                
                // Self connection (central point)
                cd.rows.push_back(i);
                cd.columns.push_back(i % N);
                cd.values.push_back(-6.0/norm);
                
                // Neighbors in the x-direction (x±1)
                if (x > 0) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(Piro::math_operations::index(x-1, y, z, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                    
                }
                if (x < n[0] - 1) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(Piro::math_operations::index(x+1, y, z, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                
                }

                // Neighbors in the y-direction (y±1)
                if (y > 0) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(Piro::math_operations::index(x, y-1, z, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                    
                }
                if (y < n[1] - 1) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(Piro::math_operations::index(x, y+1, z, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                    
                }

                // Neighbors in the z-direction (z±1)
                if (z > 0) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(Piro::math_operations::index(x, y, z-1, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                    
                }
                if (z < n[2] - 1) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(Piro::math_operations::index(x, y, z+1, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                }
                
                if (Piro::bc::isBoundaryPoint(x, y, z)) {
                    auto it = std::find(cd.columns.begin(),
                    cd.columns.end(), i);
    
                    if (it == cd.columns.end()) {
                        // Add the boundary point if not already in the matrix.
                        cd.rows.push_back(i); // for diagonal entry
                        cd.columns.push_back(i % N); // for diagonal entry
                        cd.values.push_back(10.0); // large value to enforce the BC
                    }
                    else{
                        size_t index = std::distance(cd.columns.begin(), it);
                        cd.values[index] = 10.0;
                    }
                }

                // cd.rowpointers[i + 1] = cd.columns.size();
            }
        }
    }
}

void Piro::COO::stencils::laplacian_uw(Piro::CellData& cd, std::vector<uint> n, float norm){
    int N = n[0] * n[1] * n[2];
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                
                // Self connection (central point)
                cd.rows.push_back(i);
                cd.columns.push_back(i % N);
                cd.values.push_back(-6.0/norm);
                
                // Neighbors in the x-direction (x±1)
                if (x > 0) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(math_operations::index(x-1, y, z, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                    
                }
                if (x < n[0] - 1) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                
                }

                // Neighbors in the y-direction (y±1)
                if (y > 0) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(math_operations::index(x, y-1, z, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                    
                }
                if (y < n[1] - 1) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                    
                }

                // Neighbors in the z-direction (z±1)
                if (z > 0) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(math_operations::index(x, y, z-1, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                    
                }
                if (z < n[2] - 1) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                }
                
                if (Piro::bc::isBoundaryPoint(x, y, z)) {
                    auto it = std::find(cd.columns.begin(),
                    cd.columns.end(), i);
    
                    if (it == cd.columns.end()) {
                        // Add the boundary point if not already in the matrix.
                        cd.rows.push_back(i); // for diagonal entry
                        cd.columns.push_back(i % N); // for diagonal entry
                        cd.values.push_back(10.0); // large value to enforce the BC
                    }
                    else{
                        size_t index = std::distance(cd.columns.begin(), it);
                        cd.values[index] = 10.0;
                    }
                }

                // cd.rowpointers[i + 1] = cd.columns.size();
            }
        }
    }
}

void Piro::COO::stencils::laplacian_dw(Piro::CellData& cd, std::vector<uint> n, float norm){
    int N = n[0] * n[1] * n[2];
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                
                // Self connection (central point)
                cd.rows.push_back(i);
                cd.columns.push_back(i % N);
                cd.values.push_back(-6.0/norm);
                
                // Neighbors in the x-direction (x±1)
                if (x > 0) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                    
                }
                if (x < n[0] - 1) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(math_operations::index(x + 1, y, z, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                
                }

                // Neighbors in the y-direction (y±1)
                if (y > 0) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                    
                }
                if (y < n[1] - 1) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(math_operations::index(x, y + 1, z, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                    
                }

                // Neighbors in the z-direction (z±1)
                if (z > 0) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                    
                }
                if (z < n[2] - 1) {
                    cd.rows.push_back(i);
                    cd.columns.push_back(math_operations::index(x, y, z + 1, n[0], n[1]) % N);
                    cd.values.push_back(1.0/norm);
                }
                
                if (Piro::bc::isBoundaryPoint(x, y, z)) {
                    auto it = std::find(cd.columns.begin(),
                    cd.columns.end(), i);
    
                    if (it == cd.columns.end()) {
                        // Add the boundary point if not already in the matrix.
                        cd.rows.push_back(i); // for diagonal entry
                        cd.columns.push_back(i % N); // for diagonal entry
                        cd.values.push_back(10.0); // large value to enforce the BC
                    }
                    else{
                        size_t index = std::distance(cd.columns.begin(), it);
                        cd.values[index] = 10.0;
                    }
                }

                // cd.rowpointers[i + 1] = cd.columns.size();
            }
        }
    }
}

void Piro::HT::stencils::laplacian_cd(Piro::CellData& cd, std::vector<uint> n, float norm){
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                
                // Self connection (central point)
                cd.columns.push_back(i);
                cd.values.push_back(-6.0/norm);
                
                // Neighbors in the x-direction (x±1)
                if (x > 0) {
                    cd.columns.push_back(math_operations::index(x-1, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }
                if (x < n[0] - 1) {
                    cd.columns.push_back(math_operations::index(x+1, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                
                }

                // Neighbors in the y-direction (y±1)
                if (y > 0) {
                    cd.columns.push_back(math_operations::index(x, y-1, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }
                    if (y < n[1] - 1) {
                        cd.columns.push_back(math_operations::index(x, y+1, z, n[0], n[1]));
                        cd.values.push_back(1.0/norm);
                        
                    }

                    // Neighbors in the z-direction (z±1)
                    if (z > 0) {
                        cd.columns.push_back(math_operations::index(x, y, z-1, n[0], n[1]));
                        cd.values.push_back(1.0/norm);
                        
                    }
                    if (z < n[2] - 1) {
                        cd.columns.push_back(math_operations::index(x, y, z+1, n[0], n[1]));
                        cd.values.push_back(1.0/norm);
                    }
                
                    if (Piro::bc::isBoundaryPoint(x, y, z)) {
                        auto it = std::find(cd.columns.begin(),
                        cd.columns.end(), i);
        
                        if (it == cd.columns.end()) {
                            // Add the boundary point if not already in the matrix.
                            cd.columns.push_back(i); // for diagonal entry
                            cd.values.push_back(10.0); // large value to enforce the BC
                        }
                        else{
                            size_t index = std::distance(cd.columns.begin(), it);
                            cd.values[index] = 10.0;
                        }
                    }

                cd.rowpointers[i + 1] = cd.columns.size();
            }
        }
    }
}

void Piro::HT::stencils::laplacian_uw(Piro::CellData& cd, std::vector<uint> n, float norm){
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                
                // Self connection (central point)
                cd.columns.push_back(i);
                cd.values.push_back(-6.0/norm);
                
                // Neighbors in the x-direction (x±1)
                if (x > 0) {
                    cd.columns.push_back(math_operations::index(x-1, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }
                if (x < n[0] - 1) {
                    cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                
                }

                // Neighbors in the y-direction (y±1)
                if (y > 0) {
                    cd.columns.push_back(math_operations::index(x, y-1, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }
                    if (y < n[1] - 1) {
                        cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]));
                        cd.values.push_back(1.0/norm);
                        
                    }

                    // Neighbors in the z-direction (z±1)
                    if (z > 0) {
                        cd.columns.push_back(math_operations::index(x, y, z-1, n[0], n[1]));
                        cd.values.push_back(1.0/norm);
                        
                    }
                    if (z < n[2] - 1) {
                        cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]));
                        cd.values.push_back(1.0/norm);
                    }
                
                    if (Piro::bc::isBoundaryPoint(x, y, z)) {
                        auto it = std::find(cd.columns.begin(),
                        cd.columns.end(), i);
        
                        if (it == cd.columns.end()) {
                            // Add the boundary point if not already in the matrix.
                            cd.columns.push_back(i); // for diagonal entry
                            cd.values.push_back(10.0); // large value to enforce the BC
                        }
                        else{
                            size_t index = std::distance(cd.columns.begin(), it);
                            cd.values[index] = 10.0;
                        }
                    }

                cd.rowpointers[i + 1] = cd.columns.size();
            }
        }
    }
}

void Piro::HT::stencils::laplacian_dw(Piro::CellData& cd, std::vector<uint> n, float norm){
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                
                // Self connection (central point)
                cd.columns.push_back(i);
                cd.values.push_back(-6.0/norm);
                
                // Neighbors in the x-direction (x±1)
                if (x > 0) {
                    cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }
                if (x < n[0] - 1) {
                    cd.columns.push_back(math_operations::index(x + 1, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                
                }

                // Neighbors in the y-direction (y±1)
                if (y > 0) {
                    cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]));
                    cd.values.push_back(1.0/norm);
                    
                }
                    if (y < n[1] - 1) {
                        cd.columns.push_back(math_operations::index(x, y + 1, z, n[0], n[1]));
                        cd.values.push_back(1.0/norm);
                        
                    }

                    // Neighbors in the z-direction (z±1)
                    if (z > 0) {
                        cd.columns.push_back(math_operations::index(x, y, z, n[0], n[1]));
                        cd.values.push_back(1.0/norm);
                        
                    }
                    if (z < n[2] - 1) {
                        cd.columns.push_back(math_operations::index(x, y, z + 1, n[0], n[1]));
                        cd.values.push_back(1.0/norm);
                    }
                
                    if (Piro::bc::isBoundaryPoint(x, y, z)) {
                        auto it = std::find(cd.columns.begin(),
                        cd.columns.end(), i);
        
                        if (it == cd.columns.end()) {
                            // Add the boundary point if not already in the matrix.
                            cd.columns.push_back(i); // for diagonal entry
                            cd.values.push_back(10.0); // large value to enforce the BC
                        }
                        else{
                            size_t index = std::distance(cd.columns.begin(), it);
                            cd.values[index] = 10.0;
                        }
                    }

                cd.rowpointers[i + 1] = cd.columns.size();
            }
        }
    }
}   

void Piro::CSR::stencils::div_uw_x(Piro::CellData& cd, std::vector<uint> n, float norm){Piro::CSR::stencils::gradient_uw_x(cd, n, norm);}
void Piro::CSR::stencils::div_uw_y(Piro::CellData& cd, std::vector<uint> n, float norm){Piro::CSR::stencils::gradient_uw_y(cd, n, norm);}
void Piro::CSR::stencils::div_uw_z(Piro::CellData& cd, std::vector<uint> n, float norm){Piro::CSR::stencils::gradient_uw_z(cd, n, norm);}
void Piro::CSR::stencils::div_cd_x(Piro::CellData& cd, std::vector<uint> n, float norm){Piro::CSR::stencils::gradient_cd_x(cd, n, norm);}
void Piro::CSR::stencils::div_cd_y(Piro::CellData& cd, std::vector<uint> n, float norm){Piro::CSR::stencils::gradient_cd_y(cd, n, norm);}
void Piro::CSR::stencils::div_cd_z(Piro::CellData& cd, std::vector<uint> n, float norm){Piro::CSR::stencils::gradient_cd_z(cd, n, norm);}
void Piro::CSR::stencils::div_dw_x(Piro::CellData& cd, std::vector<uint> n, float norm){Piro::CSR::stencils::gradient_dw_x(cd, n, norm);}
void Piro::CSR::stencils::div_dw_y(Piro::CellData& cd, std::vector<uint> n, float norm){Piro::CSR::stencils::gradient_dw_y(cd, n, norm);}
void Piro::CSR::stencils::div_dw_z(Piro::CellData& cd, std::vector<uint> n, float norm){Piro::CSR::stencils::gradient_dw_z(cd, n, norm);}