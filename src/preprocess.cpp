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

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#elif _WIN32
    #include "../dependencies/include/CL/opencl.h"
#else
    #include "../dependencies/include/CL/opencl.h"
#endif

Giro::MeshParams MP;
Giro::SolveParams SP;
Giro::DeviceParams DP;
Giro::CellDataGPU CDGPU;
bool LAP_INIT = false;
int ts = 0;

// Function to map 3D indices to 1D
int idx(int i, int j, int k, int N_x, int N_y) {
    return i + j * N_x + k * N_x * N_y;
}

int index(int i, int j, int k, int Nx, int Ny) {
    return i + Nx * (j + Ny * k);
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
    // number of scalars defined in the .ini file
    MP.scalarnum = countSpaces(reader.get("Simulation", "Scalars", "default_value")) + 1;
    // number of vectors defined in the .ini file
    MP.vectornum = countWords(reader.get("Simulation", "Vectors", ""));
    // Initial condition type.
    MP.ICtype = convertStringVectorToInt(splitString(reader.get("IC", "type", "default_value"), ' '));
    // File location needed if ICtype == 2
    MP.ICfiles = splitString(reader.get("IC", "filename", "default_value"), ' ');
    std::cout << "Initialising scalars and vectors" << std::endl;
    int j = 0;
    Giro::CellData CD;
    CLBuffer CD_GPU;
    // total number of cells
    int N = MP.n[0] * MP.n[1] * MP.n[2];
    std::cout << "Total number of cells : " << N << std::endl;
    for (int i = 0; i < MP.scalarnum; i++){
        
        CD.Scalars = MP.scalarlist[i];
        MP.AMR[0].CD.push_back(CD);
        MP.AMR[0].CD[i].type = 0;
        MP.AMR[0].CD[i].values = initialcondition(i, MP.AMR[0].CD[i].type);
        // push scalar data to gpu
        CDGPU.values_gpu.push_back(CD_GPU);
        CDGPU.values_gpu[i].buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * N, MP.AMR[0].CD[i].values.data(), &err);
        j += 1;
    }
    for (int i = j; i < j + MP.vectornum; i++){
        
        CD.Scalars = MP.vectorlist[i - j];
        MP.AMR[0].CD.push_back(CD);
        MP.AMR[0].CD[i].type = 1;
        MP.AMR[0].CD[i].values = initialcondition(i, MP.AMR[0].CD[i].type);
        // push vector data to gpu
        CDGPU.values_gpu.push_back(CD_GPU);
        CDGPU.values_gpu[i].buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * N, MP.AMR[0].CD[i].values.data(), &err);
        
    }
    std::cout << "Initialising scalars and vectors completed!" << std::endl;
    SP.delta[0] = MP.l[0] / float(MP.n[0] - 2);
    SP.delta[1] = MP.l[1] / float(MP.n[1] - 2);
    SP.delta[2] = MP.l[2] / float(MP.n[2] - 2);

    SP.timestep = std::stof(reader.get("Solve", "Timestep", "default_value"));
    SP.totaltime = std::stof(reader.get("Solve", "TotalTime", "default_value"));
    SP.save = std::stoi(reader.get("Solve", "Save", "default_value"));
    SP.totaltimesteps = SP.totaltime / SP.timestep;
    
    readbc();

    return 0;
}

int laplacian_CSR_init(){
    int N = MP.n[0] * MP.n[1] * MP.n[2];
    Giro::CellData CD;
    CLBuffer CD_GPU;
    MP.AMR[0].CD.push_back(CD);
    CDGPU.laplacian_csr.push_back(CD_GPU);
    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].type = 2; // row pointers
    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].rowpointers.assign(N, 0.0);
    MP.AMR[0].CD.push_back(CD);
    CDGPU.laplacian_csr.push_back(CD_GPU);
    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].type = 3; // columns
    // MP.AMR[0].CD[MP.vectornum + MP.scalarnum].columns.assign(N, 0.0);
    MP.AMR[0].CD.push_back(CD);
    CDGPU.laplacian_csr.push_back(CD_GPU);
    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].type = 4; // values
    // MP.AMR[0].CD[MP.vectornum + MP.scalarnum].values.assign(N, 0.0);
    // printVector(MP.AMR[0].CD[MP.vectornum + MP.scalarnum].rowpointers);
    // Iterate over all grid points
    int count = 0;
    for (int z = 0; z < MP.n[2]; ++z) {
        for (int y = 0; y < MP.n[1]; ++y) {
            for (int x = 0; x < MP.n[0]; ++x) {
                int i = index(x, y, z, MP.n[0], MP.n[1]); // 1D index of the (x, y, z) point
                
                // Self connection (central point)
                MP.AMR[0].CD[MP.vectornum + MP.scalarnum].columns.push_back(i);
                MP.AMR[0].CD[MP.vectornum + MP.scalarnum].values.push_back(-6.0);

                // Neighbors in the x-direction (x±1)
                if (x > 0) {
                    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].columns.push_back(index(x-1, y, z, MP.n[0], MP.n[1]));
                    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].values.push_back(1.0);
                    count++;
                }
                if (x < MP.n[0] - 1) {
                    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].columns.push_back(index(x+1, y, z, MP.n[0], MP.n[1]));
                    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].values.push_back(1.0);
                    count++;
                }

                // Neighbors in the y-direction (y±1)
                if (y > 0) {
                    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].columns.push_back(index(x, y-1, z, MP.n[0], MP.n[1]));
                    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].values.push_back(1.0);
                    count++;
                }
                if (y < MP.n[1] - 1) {
                    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].columns.push_back(index(x, y+1, z, MP.n[0], MP.n[1]));
                    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].values.push_back(1.0);
                    count++;
                }

                // Neighbors in the z-direction (z±1)
                if (z > 0) {
                    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].columns.push_back(index(x, y, z-1, MP.n[0], MP.n[1]));
                    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].values.push_back(1.0);
                    count++;
                }
                if (z < MP.n[2] - 1) {
                    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].columns.push_back(index(x, y, z+1, MP.n[0], MP.n[1]));
                    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].values.push_back(1.0);
                    count++;
                }

                MP.AMR[0].CD[MP.vectornum + MP.scalarnum].rowpointers[i] = count; // Starting index in the column/MP.AMR[0].CD[MP.vectornum + MP.scalarnum + 2] array

            }
        }
    }

    //for (int i = 1; i <= N; ++i) {
    //    MP.AMR[0].CD[MP.vectornum + MP.scalarnum].rowpointers[i] += MP.AMR[0].CD[MP.vectornum + MP.scalarnum].rowpointers[i - 1];
    //}
    CDGPU.laplacian_csr[0].buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * MP.AMR[0].CD[MP.vectornum + MP.scalarnum].rowpointers.size(), MP.AMR[0].CD[MP.vectornum + MP.scalarnum].rowpointers.data(), &err);

    CDGPU.laplacian_csr[1].buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) *  MP.AMR[0].CD[MP.vectornum + MP.scalarnum].columns.size(), MP.AMR[0].CD[MP.vectornum + MP.scalarnum].columns.data(), &err);

    CDGPU.laplacian_csr[2].buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) *  MP.AMR[0].CD[MP.vectornum + MP.scalarnum].values.size(), MP.AMR[0].CD[MP.vectornum + MP.scalarnum].values.data(), &err);
    
    LAP_INIT = true;
    // printVector(MP.AMR[0].CD[MP.vectornum + MP.scalarnum].rowpointers);
    // printVector(MP.AMR[0].CD[MP.vectornum + MP.scalarnum].columns);
    // printVector(MP.AMR[0].CD[MP.vectornum + MP.scalarnum].values);
    // std::cout << MP.AMR[0].CD[MP.vectornum + MP.scalarnum].values.size() << std::endl;
    return 0;
}

