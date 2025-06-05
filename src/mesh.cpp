#include <mesh.hpp>
#include <fileutilities.hpp>
#include <datatypes.hpp>
#include <printutilities.hpp>
#include <vector>
#include <logger.hpp>
#include <octree.hpp>
std::vector<std::vector<int>> Piro::mesh_operations::readmesh::readlevels(){
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    int level = MP.getvalue<int>(Piro::MeshParams::LEVELS);
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    Piro::logger::info("Reading mesh : ", MP.getvalue<std::vector<std::string>>(Piro::MeshParams::MESHTYPE)[0]);
    std::vector<std::vector<int>> pm = Piro::file_utilities::IniReader::parsemesh(Piro::file_utilities::current_path.string() 
                                        + "/assets/Mesh/" 
                                        + MP.getvalue<std::vector<std::string>>(Piro::MeshParams::MESHTYPE)[0] 
                                        + "/level.txt");
    std::vector<int> levelcount(level + 1);
    int total_cells = 0;
    // get the number of levels
    levelcount[0] = n[0] * n[1] * n[2];
    for(int pml = 0; pml < pm.size(); pml++){
        if(pml != 0){
             levelcount[pm[pml][0]]++;
        }
        // Piro::logger::info(pm[pml][0]);
    }
    for(int pml = 0; pml < level; pml++){
        if(pml != 0){
            int coarsecell = (pm[pml][3] - pm[pml][2])*(pm[pml][5] - pm[pml][4])*(pm[pml][7] - pm[pml][6]);
            levelcount[pml] *= coarsecell * 8 * std::min(pml, 1);
        }
        total_cells += levelcount[pml];
        Piro::logger::info("Level", pml, " : ", levelcount[pml]);
    }
    // Piro::print_utilities::printVector(levelcount);
    Piro::logger::info("Total cells : ", total_cells);
    MP.setvalue(Piro::MeshParams::LEVELCOUNT, levelcount);
    MP.setvalue(Piro::MeshParams::TOTALCELLS, total_cells);
    MP.setvalue(Piro::MeshParams::MESH, pm);
    // Piro::print_utilities::printVector(pm[0]);
    // std::vector<std::vector<float>> tempcoord = Piro::mesh_operations::octree::get_coord(pm[0]);
    // Piro::print_utilities::printMatrix(tempcoord);
    // std::exit(1);
    return pm;
    
}