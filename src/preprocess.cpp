#include <iostream>
#include <vector>
#include <string>

#include "../dependencies/include/preprocess.hpp"
#include "../dependencies/include/init.hpp"
#include "../dependencies/include/inireader.hpp"
#include "../dependencies/include/datatypes.hpp"
#include "../dependencies/include/extras.hpp"
#include "../dependencies/include/solve.hpp"
#include "../dependencies/include/ic.hpp"
#include "../dependencies/include/bc.hpp"

Giro::MeshParams MP;
Giro::SolveParams SP;
Giro::DeviceParams DP;
std::vector<std::vector<float>> scagradmatrix, scadivmatrix, vecmatrix;
int ts = 0;
float* scalapvectorpointer;

// Function to map 3D indices to 1D
int idx(int i, int j, int k, int N_x, int N_y) {
    return i + j * N_x + k * N_x * N_y;
}

int preprocess(const std::string& name) {
    
    std::cout << "Preprocess step initiated" << std::endl;
    std::cout << "Setup file : " << name << std::endl;
    IniReader reader(current_path.string() + "/assets/" + name);
    // Print all sections and key-value pairs
    // reader.print();
    DP.id = std::stoi(reader.get("Device", "id", "default_value"));
    DP.type = std::stoi(reader.get("Device", "type", "default_value"));
    DP.platformid = std::stoi(reader.get("Device", "platformid", "default_value"));

    SP.casename = reader.get("File", "casename", "default_value");
    SP.restart = std::stoi(reader.get("File", "restart", "default_value"));
    SP.timescheme = std::stoi(reader.get("Schemes", "Time", "default_value"));
    SP.spacescheme = std::stoi(reader.get("Schemes", "Space", "default_value"));

    MP.o = convertStringVectorToFloat(splitString(reader.get("Mesh", "o", "default_value"), ' '));
    MP.s = convertStringVectorToFloat(splitString(reader.get("Mesh", "s", "default_value"), ' '));
    MP.n = convertStringVectorToUInt(splitString(reader.get("Mesh", "n", "default_value"), ' '));
    MP.l = convertStringVectorToFloat(splitString(reader.get("Mesh", "l", "default_value"), ' '));
    MP.index = convertStringVectorToInt(splitString(reader.get("Mesh", "index", "default_value"), ' '));
    MP.constantslist = splitString(reader.get("Simulation", "Constants", "default_value"), ' ');
    MP.constantsvalues = convertStringVectorToFloat(splitString(reader.get("Simulation", "Values", "default_value"), ' '));
    MP.scalarlist = splitString(reader.get("Simulation", "Scalars", "default_value"), ' ');
    MP.vectorlist = splitString(reader.get("Simulation", "Vectors", "default_value"), ' ');
    MP.meshtype = std::stoi(reader.get("Mesh", "MeshType", "default_value"));
    MP.levels = std::stoi(reader.get("Mesh", "levels", "default_value"));

    MP.n[0] += 2;
    MP.n[1] += 2;
    MP.n[2] += 2;

    init();
    opencl_init();
    opencl_build();
    
    for(int i = 0; i <= MP.levels; i++){

        Giro::AMR AMR;
        AMR.WholeExtent[0] = 0;
        AMR.WholeExtent[1] = MP.n[0];
        AMR.WholeExtent[2] = 0;
        AMR.WholeExtent[3] = MP.n[1];
        AMR.WholeExtent[4] = 0;
        AMR.WholeExtent[5] = MP.n[2];

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
    MP.ICtype = convertStringVectorToInt(splitString(reader.get("IC", "type", "default_value"), ' '));
    MP.ICfiles = splitString(reader.get("IC", "filename", "default_value"), ' ');
    std::cout << "Initialising scalars and vectors" << std::endl;
    int j = 0;
    Giro::CellData CD;
    for (int i = 0; i < MP.scalarnum; i++){
        
        CD.Scalars = MP.scalarlist[i];
        MP.AMR[0].CD.push_back(CD);
        MP.AMR[0].CD[i].type = 0;
        MP.AMR[0].CD[i].values = initialcondition(i, MP.AMR[0].CD[i].type, MP.ICtype[i]);
        
        j += 1;
    }
    for (int i = j; i < j + MP.vectornum; i++){
        
        CD.Scalars = MP.vectorlist[i - j];
        MP.AMR[0].CD.push_back(CD);
        MP.AMR[0].CD[i].type = 1;
        MP.AMR[0].CD[i].values = initialcondition(i, MP.AMR[0].CD[i].type, MP.ICtype[i]);
        
    }
    std::cout << "Initialising scalars and vectors completed!" << std::endl;
    SP.delta[0] = MP.l[0] / float(MP.n[0] - 2);
    SP.delta[1] = MP.l[1] / float(MP.n[1] - 2);
    SP.delta[2] = MP.l[2] / float(MP.n[2] - 2);

    SP.timestep = std::stof(reader.get("Solve", "Timestep", "default_value"));
    SP.totaltime = std::stof(reader.get("Solve", "TotalTime", "default_value"));
    SP.totaltimesteps = SP.totaltime / SP.timestep;
    

    
    readbc();

    return 0;
}