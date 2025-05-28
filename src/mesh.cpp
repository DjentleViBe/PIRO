#include <mesh.hpp>
#include <fileutilities.hpp>
#include <datatypes.hpp>
#include <printutilities.hpp>
#include <vector>

std::vector<std::vector<int>> Piro::mesh_operations::readmesh::readlevels(int level){
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    
    std::vector<std::vector<int>> pm = Piro::file_utilities::IniReader::parsemesh(Piro::file_utilities::current_path.string() 
                                        + "/assets/Mesh/" 
                                        + MP.getvalue<std::vector<std::string>>(Piro::MeshParams::MESHTYPE)[0] 
                                        + "/level" + std::to_string(level) + ".txt");
    Piro::print_utilities::printMatrix(pm);
    return pm;
    
}