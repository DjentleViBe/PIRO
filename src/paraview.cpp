#include "../dependencies/include/postprocess.hpp"
#include "../dependencies/include/datatypes.hpp"
#include "../dependencies/include/init.hpp"
#include "../dependencies/include/extras.hpp"
#include <string>

void writevtk(){
    writefile(current_path.string() + "/mesh/mesh.vtk", "abc");
}