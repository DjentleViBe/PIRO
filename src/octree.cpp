#include <mesh.hpp>
#include <vector>
#include <octree.hpp>
#include <cmath>
#include <datatypes.hpp>
#include <printutilities.hpp>
#include <logger.hpp>

int Piro::mesh_operations::octree::findind(std::vector<std::vector<int>> pm, int match, int level){
    for(int k = 0; k < pm.size(); k++){
        if(pm[k][1] == match && pm[k][0] == level - 1){
            return k;
        }
    }
    Piro::logger::info("index error - Octree mesh, match =", match, ", level =" , level - 1);
    Piro::print_utilities::printMatrix(pm);
    return -1;
}

std::vector<float> Piro::mesh_operations::octree::get_ijk(std::vector<int> index, std::vector<float>& coord, std::vector<std::vector<int>> pm){
    int level = index[0];
    std::vector<float> i_val(3);
    if (level <= 1){
        i_val[0] = static_cast<float>(index[2]) / 2.0;
        i_val[1] = static_cast<float>(index[4]) / 2.0;
        i_val[2] = static_cast<float>(index[6]) / 2.0;
    }
    else{
        int match = Piro::mesh_operations::octree::findind(pm, index[1], level);
        std::vector<float> parent_coord = {0.0f, 0.0f, 0.0f};
        std::vector<float> parent_val = Piro::mesh_operations::octree::get_ijk(pm[match], parent_coord, pm);

        for(int j = 0; j < 3; j++){
            parent_coord[j] += parent_val[j] / (pow(2, pm[match][0] - 1));
        }
        i_val[0] = static_cast<float>(index[2] / 2.0) - parent_coord[0] * (pow(2, (level - 1)));
        i_val[1] = static_cast<float>(index[4] / 2.0) - parent_coord[1] * (pow(2, (level - 1)));
        i_val[2] = static_cast<float>(index[6] / 2.0) - parent_coord[2] * (pow(2, (level - 1)));
    }
    for(int i = 0; i < std::min(coord.size(), i_val.size()); i++){
        coord[i] += static_cast<float>(i_val[i]) / (pow(2, (level - 1)));
    }
    return i_val;
}

std::vector<std::vector<float>> Piro::mesh_operations::octree::get_origin(std::vector<int> ind){
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    std::vector<float> l = MP.getvalue<std::vector<float>>(Piro::MeshParams::L);
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<float> delta(3);
    int cellcount = MP.getvalue<std::vector<int>>(Piro::MeshParams::LEVELCOUNT)[ind[0]];
    std::vector<std::vector<float>> origin_collect(cellcount, std::vector<float>(3, 0.0));
    delta[0] = (l[0] / (float)(n[0] - 2)) / std::pow(2, ind[0]);
    delta[1] = (l[1] / (float)(n[1] - 2)) / std::pow(2, ind[0]);
    delta[2] = (l[2] / (float)(n[2] - 2)) / std::pow(2, ind[0]);
    // Piro::print_utilities::printVector(delta);
    // Piro::logger::info("Printing coordinates");
    int b = 0;
    for(int w = ind[2]; w < ind[3]; w++){
        for(int t = ind[4]; t < ind[5]; t++){
            for(int f = ind[6]; f < ind[7]; f++){
                if(ind[0] == 0){
                    std::vector<int> indtemp = {ind[0], ind[1], w, w + 1, t, t + 1, f, f + 1};
                    std::vector<float> coord = {0.0f, 0.0f, 0.0f};
                    get_ijk(indtemp, coord, MP.getvalue<std::vector<std::vector<int>>>(Piro::MeshParams::MESH));
                    origin_collect[b] = {coord[0] + (0.5f * delta[0]), coord[1] + (0.5f * delta[1]), coord[2] + (0.5f * delta[2])};
                    // Piro::logger::info(coord[0] + 0.5 * delta[0], coord[1] + 0.5 * delta[1], coord[2] + 0.5 * delta[2]);
                }
                else{
                    std::vector<int> indtemp = {ind[0], ind[1], w, w + 1, t, t + 1, f, f + 1};
                    std::vector<float> coord = {0.0f, 0.0f, 0.0f};
                    get_ijk(indtemp, coord, MP.getvalue<std::vector<std::vector<int>>>(Piro::MeshParams::MESH));
                    origin_collect[b++]   = {coord[0] + (0.5f * delta[0]), coord[1] + (0.5f * delta[1]), coord[2] + (0.5f * delta[2])};
                    origin_collect[b++] = {coord[0] + delta[0] + (0.5f * delta[0]), coord[1] + (0.5f * delta[1]), coord[2] + (0.5f * delta[2])};
                    origin_collect[b++] = {coord[0] + (0.5f * delta[0]), coord[1] + delta[0] + (0.5f * delta[1]), coord[2] + (0.5f * delta[2])};
                    origin_collect[b++] = {coord[0] + delta[0] + (0.5f * delta[0]), coord[1] + delta[1] + (0.5f * delta[1]), coord[2] + (0.5f * delta[2])};
                    
                    origin_collect[b++] = {coord[0] + (0.5f * delta[0]), coord[1] + (0.5f * delta[1]), coord[2] + delta[2] + (0.5f * delta[2])};
                    origin_collect[b++] = {coord[0] + delta[0] + (0.5f * delta[0]), coord[1] + (0.5f * delta[1]), coord[2] + delta[2] + (0.5f * delta[2])};
                    origin_collect[b++] = {coord[0] + (0.5f * delta[0]), coord[1] + delta[1] + (0.5f * delta[1]), coord[2] + delta[2] + (0.5f * delta[2])};
                    origin_collect[b++] = {coord[0] + delta[0] + (0.5f * delta[0]), coord[1] + delta[1] + (0.5f * delta[1]), coord[2] + delta[2]+ (0.5f * delta[2])};

                }
                b++;
            }
        }
    }
    // Piro::print_utilities::printVector(coord);
    return origin_collect;

}