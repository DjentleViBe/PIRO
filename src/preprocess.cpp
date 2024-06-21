#include <iostream>
#include <vector>
#include "../dependencies/include/preprocess.hpp"
#include "../dependencies/include/inireader.hpp"
#include "../dependencies/include/datatypes.hpp"
#include "../dependencies/include/extras.hpp"

Giro::MeshParams MP;

std::vector<float> initialize(int type){
    std::vector<float> values;
    if(type == 0){
        values.assign(MP.n[0] * MP.n[1] * MP.n[2], 0.0);
    }
    return values;
}

int preprocess() {
    std::cout << "This is the preprocess source" << std::endl;

    IniReader reader("setup.ini");
    // Print all sections and key-value pairs
    // reader.print();
    MP.o = convertStringVectorToFloat(splitString(reader.get("Mesh", "o", "default_value"), ' '));
    MP.s = convertStringVectorToFloat(splitString(reader.get("Mesh", "s", "default_value"), ' '));
    MP.n = convertStringVectorToInt(splitString(reader.get("Mesh", "n", "default_value"), ' '));
    MP.l = convertStringVectorToInt(splitString(reader.get("Mesh", "l", "default_value"), ' '));
    MP.index = convertStringVectorToInt(splitString(reader.get("Mesh", "index", "default_value"), ' '));
    MP.scalarlist = splitString(reader.get("Simulation", "Scalars", "default_value"), ' ');
    MP.vectorlist = splitString(reader.get("Simulation", "Vectors", "default_value"), ' ');
    
    MP.meshtype = std::stoi(reader.get("Mesh", "MeshType", "default_value"));
    MP.levels = std::stoi(reader.get("Mesh", "levels", "default_value"));

    for(int i = 0; i <= MP.levels; i++){

        Giro::AMR AMR;
        AMR.WholeExtent[0] = 0;
        AMR.WholeExtent[1] = MP.l[0];
        AMR.WholeExtent[2] = 0;
        AMR.WholeExtent[3] = MP.l[1];
        AMR.WholeExtent[4] = 0;
        AMR.WholeExtent[5] = MP.l[2];

        AMR.Origin[0] = MP.o[0];
        AMR.Origin[1] = MP.o[1];
        AMR.Origin[2] = MP.o[2];

        AMR.Spacing[0] = MP.s[0];
        AMR.Spacing[1] = MP.s[1];
        AMR.Spacing[2] = MP.s[2];
        
        MP.AMR.push_back(AMR);
        
    }

    MP.scalarnum = countSpaces(reader.get("Simulation", "Scalars", "default_value")) + 1;
    MP.vectornum = countSpaces(reader.get("Simulation", "Vectors", "default_value")) + 1;
    MP.ICtype = std::stoi(reader.get("IC", "type", "default_value"));
    int j = 0;
    for (int i = 0; i < MP.scalarnum; i++){
        Giro::CellData CD;
        CD.Scalars = MP.scalarlist[i];
        MP.AMR[0].CD.push_back(CD);
        MP.AMR[0].CD[i].values = initialize(MP.ICtype);
        
        j += 1;
    }
    for (int i = j - 1; i < j + MP.vectornum; i++){
        Giro::CellData CD;
        CD.Scalars = MP.vectorlist[i];
        MP.AMR[0].CD.push_back(CD);
        MP.AMR[0].CD[i].values = initialize(MP.ICtype);
        std::cout << MP.AMR[0].CD[i].Scalars << std::endl;
    }
    
    return 0;
}