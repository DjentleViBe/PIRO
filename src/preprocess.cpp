#include <iostream>
#include <vector>
#include "../dependencies/include/preprocess.hpp"
#include "../dependencies/include/inireader.hpp"
#include "../dependencies/include/datatypes.hpp"
#include "../dependencies/include/extras.hpp"

Giro::MeshParams MP;
Giro::SolveParams SP;
std::vector<std::vector<float>> scadivmatrix, vecdivmatrix, scalapmatrix, veclapmatrix;
                
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
        // std::cout << MP.AMR[0].CD[i].Scalars << std::endl;
    }

    SP.delta[0] = MP.l[0] / MP.n[0];
    SP.delta[1] = MP.l[1] / MP.n[1];
    SP.delta[2] = MP.l[2] / MP.n[2];

    SP.deltaT = std::stof(reader.get("Solve", "Timestep", "default_value"));

    // Generate scalar and vector divergence matrix for the cells

    
    // Geenrate scalar and vector laplacian matrix for the cells
    // Resize the outer vector to have 3 rows
    scalapmatrix.resize(MP.n[0]*MP.n[1]*MP.n[2]);
    veclapmatrix.resize(MP.n[0]*MP.n[1]*MP.n[2]);

    // Resize each inner vector to have 4 columns and initialize elements to 0.0f
    for (size_t i = 0; i < scalapmatrix.size(); ++i) {
        scalapmatrix[i].resize(MP.n[0]*MP.n[1]*MP.n[2], 0.0f);
        veclapmatrix[i].resize(MP.n[0]*MP.n[1]*MP.n[2], 0.0f);
    }
    float subd = SP.deltaT / (SP.delta[0] * SP.delta[0]);
    float supd = SP.deltaT / (SP.delta[0] * SP.delta[0]);
    float ds = -6.0 * SP.deltaT / (SP.delta[0] * SP.delta[0]);
    float dv1 = -2.0 * SP.deltaT / (SP.delta[0] * SP.delta[0]);

    // for different ratios of cells in Y and Z directions
    //float dv2 = -2.0 * SP.deltaT / (SP.delta[1] * SP.delta[1]);
    //float dv3 = -2.0 * SP.deltaT / (SP.delta[2] * SP.delta[2]);
    // Set the main diagonal (index 0)
    for (int i = 0; i < MP.n[0]*MP.n[1]*MP.n[2]; ++i) {
        scalapmatrix[i][i] = ds;  // 1.0 or any other desired value
        veclapmatrix[i][i] = dv1;
    }

    // Set the subdiagonal (index -1)
    for (int i = 1; i < MP.n[0]*MP.n[1]*MP.n[2]; ++i) {
        scalapmatrix[i][i - 1] = subd;  // -1.0 or any other desired value
        veclapmatrix[i][i - 1] = subd; 
    }

    // Set the superdiagonal (index +1)
    for (int i = 0; i < MP.n[0]*MP.n[1]*MP.n[2] - 1; ++i) {
        scalapmatrix[i][i + 1] = supd;  // -1.0 or any other desired value
        veclapmatrix[i][i + 1] = supd; 
    }
    
    return 0;
}