#include <iostream>
#include "../dependencies/include/preprocess.hpp"
#include "../dependencies/include/inireader.hpp"
#include "../dependencies/include/datatypes.hpp"

Giro::MeshParams MP;

int preprocess() {
    std::cout << "This is the preprocess source" << std::endl;

    IniReader reader("setup.ini");
    // Print all sections and key-value pairs
    // reader.print();
    
    MP.meshtype = std::stoi(reader.get("PreProcess", "MeshType", "default_value"));
    MP.n[0] = std::stoi(reader.get("PreProcess", "nx", "default_value"));
    MP.n[1] = std::stoi(reader.get("PreProcess", "ny", "default_value"));
    MP.n[2] = std::stoi(reader.get("PreProcess", "nz", "default_value"));

    MP.l[0] = std::stoi(reader.get("PreProcess", "lx", "default_value"));
    MP.l[1] = std::stoi(reader.get("PreProcess", "ly", "default_value"));
    MP.l[2] = std::stoi(reader.get("PreProcess", "lz", "default_value"));

    return 0;
}