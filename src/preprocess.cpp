#include <iostream>
#include <vector>
#include <string>

#include <preprocess.hpp>
#include <init.hpp>
#include <datatypes.hpp>
#include <extras.hpp>
#include <stringutilities.hpp>
#include <ic.hpp>
#include <bc.hpp>
#include <operatoroverload.hpp>
#include <CL/opencl.h>
#include <cmath>
#include <logger.hpp>
#include <mathoperations.hpp>
#include <fileutilities.hpp>
#include <openclutilities.hpp>

Piro::Equation RHS;
Piro::CLBuffer CD_GPU;
cl_mem RHSterms;

bool LAP_INIT = false;
bool RHS_INIT = false;
int ts = 0;

int Piro::preprocess(const std::string& name) {
    
    Piro::logger::info("Preprocess step initiated");
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    Piro::DeviceParams& DP = Piro::DeviceParams::getInstance();
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::logger& logg = Piro::logger::getInstance();

    Piro::logger::info("Setup file : ",name);
    Piro::file_utilities::IniReader reader(current_path.string() + "/assets/" + name);
    // Print all sections and key-value pairs
    // reader.print();
    DP.setvalue(Piro::DeviceParams::ID, std::stoi(reader.get("Device", "id", "default_value")));
    DP.setvalue(Piro::DeviceParams::TYPE, std::stoi(reader.get("Device", "type", "default_value")));
    DP.setvalue(Piro::DeviceParams::PLATFORMID, std::stoi(reader.get("Device", "platformid", "default_value")));

    SP.setvalue(Piro::SolveParams::CASENAME, reader.get("File", "casename", "default_value"));
    SP.setvalue(Piro::SolveParams::RESTART, std::stoi(reader.get("File", "restart", "default_value")));
    SP.setvalue(Piro::SolveParams::TIMESCHEME, std::stoi(reader.get("Schemes", "Time", "default_value")));
    SP.setvalue(Piro::SolveParams::SPACESCHEME, std::stoi(reader.get("Schemes", "Space", "default_value")));
    SP.setvalue(Piro::SolveParams::SOLVERSCHEME, std::stoi(reader.get("Schemes", "Solver", "default_value")));

    MP.setvalue(Piro::MeshParams::O, Piro::string_utilities::convertStringVectorToFloat(Piro::string_utilities::splitString(reader.get("Mesh", "o", "default_value"), ' ')));
    MP.setvalue(Piro::MeshParams::S, Piro::string_utilities::convertStringVectorToFloat(Piro::string_utilities::splitString(reader.get("Mesh", "s", "default_value"), ' ')));
    MP.setvalue(Piro::MeshParams::num_cells, Piro::string_utilities::convertStringVectorToUInt(Piro::string_utilities::splitString(reader.get("Mesh", "n", "default_value"), ' ')));
    MP.setvalue(Piro::MeshParams::L, Piro::string_utilities::convertStringVectorToFloat(Piro::string_utilities::splitString(reader.get("Mesh", "l", "default_value"), ' ')));
    MP.setvalue(Piro::MeshParams::INDEX, Piro::string_utilities::convertStringVectorToInt(Piro::string_utilities::splitString(reader.get("Mesh", "index", "default_value"), ' ')));
    MP.setvalue(Piro::MeshParams::CONSTANTSLIST, Piro::string_utilities::splitString(reader.get("Simulation", "Constants", "default_value"), ' '));
    MP.setvalue(Piro::MeshParams::CONSTANTSVALUES, Piro::string_utilities::convertStringVectorToFloat(Piro::string_utilities::splitString(reader.get("Simulation", "Values", "default_value"), ' ')));
    MP.setvalue(Piro::MeshParams::SCALARLIST, Piro::string_utilities::splitString(reader.get("Simulation", "Scalars", "default_value"), ' '));
    MP.setvalue(Piro::MeshParams::VECTORLIST, Piro::string_utilities::splitString(reader.get("Simulation", "Vectors", "default_value"), ' '));
    MP.setvalue(Piro::MeshParams::MESHTYPE, std::stoi(reader.get("Mesh", "MeshType", "default_value")));
    MP.setvalue(Piro::MeshParams::LEVELS, std::stoi(reader.get("Mesh", "levels", "default_value")));
    logg.setvalue(std::stoi(reader.get("Debug", "Verbose", "default_value")));

    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<float> l = MP.getvalue<std::vector<float>>(Piro::MeshParams::L);
    std::vector<float> o = MP.getvalue<std::vector<float>>(Piro::MeshParams::O);
    std::vector<float> s = MP.getvalue<std::vector<float>>(Piro::MeshParams::S);
    
    n[0] += 2;
    n[1] += 2;
    n[2] += 2;

    MP.setvalue(Piro::MeshParams::num_cells, n);

    Piro::init();
    Piro::opencl_init();
    if(compile == 0){
        Piro::opencl_run();
    }
    else{
        Piro::opencl_build();
    }
    
    std::vector<AMR> AMR_collect;
    for(int i = 0; i <= MP.getvalue<int>(Piro::MeshParams::LEVELS); i++){

        Piro::AMR AMR_val;
        
        AMR_val.WholeExtent[0] = 0;
        AMR_val.WholeExtent[1] = n[0];
        AMR_val.WholeExtent[2] = 0;
        AMR_val.WholeExtent[3] = n[1];
        AMR_val.WholeExtent[4] = 0;
        AMR_val.WholeExtent[5] = n[2];

        AMR_val.Origin[0] = o[0];
        AMR_val.Origin[1] = o[1];
        AMR_val.Origin[2] = o[2];

        AMR_val.Spacing[0] = s[0];
        AMR_val.Spacing[1] = s[1];
        AMR_val.Spacing[2] = s[2];
        
        AMR_collect.push_back(AMR_val);
    }
    MP.setvalue(Piro::MeshParams::AMR, AMR_collect);

    // number of scalars defined in the .ini file
    MP.setvalue(Piro::MeshParams::SCALARNUM, Piro::string_utilities::countSpaces(reader.get("Simulation", "Scalars", "default_value")) + 1);
    // number of vectors defined in the .ini file
    MP.setvalue(Piro::MeshParams::VECTORNUM, Piro::string_utilities::countWords(reader.get("Simulation", "Vectors", "")));
    // Initial condition type.
    MP.setvalue(Piro::MeshParams::ICTYPE, Piro::string_utilities::convertStringVectorToInt(Piro::string_utilities::splitString(reader.get("IC", "type", "default_value"), ' ')));
    // File location needed if ICtype == 2
    MP.setvalue(Piro::MeshParams::ICFILES, Piro::string_utilities::splitString(reader.get("IC", "filename", "default_value"), ' '));
    

    Piro::logger::info("Initialising scalars and vectors");
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    int j = 0;
    Piro::CellData CD;
    cl_int err;
    //CLBuffer CD_GPU;
    // total number of cells
    int N = n[0] * n[1] * n[2];
    Piro::logger::info("Total number of cells : ", N);
    std::vector<Piro::CLBuffer> CDGPU_collect;
    for (int i = 0; i < MP.getvalue<int>(Piro::MeshParams::SCALARNUM); i++){
        
        CD.Scalars = MP.getvalue<std::vector<std::string>>(Piro::MeshParams::SCALARLIST)[i];
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD.push_back(CD);
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[i].type = 0;
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[i].values = initialcondition(i, MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[i].type);
        // push scalar data to gpu
        CDGPU_collect.push_back(CD_GPU);
        CDGPU_collect[i].buffer = clCreateBuffer(Piro::kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[i].values.data(), &err);
        j += 1;
    }
    for (int i = j; i < j + MP.getvalue<int>(Piro::MeshParams::VECTORNUM); i++){
        
        CD.Scalars = MP.getvalue<std::vector<std::string>>(Piro::MeshParams::VECTORLIST)[i - j];
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD.push_back(CD);
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[i].type = 1;
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[i].values = initialcondition(i, MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[i].type);
        // push vector data to gpu
        CDGPU_collect.push_back(CD_GPU);
        CDGPU_collect[i].buffer = clCreateBuffer(Piro::kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[i].values.data(), &err);
        
    }
    CDGPU.setvalue(Piro::CellDataGPU::VALUES_GPU, CDGPU_collect);
    Piro::logger::info("Initialising scalars and vectors completed!");
    std::vector<float> delta = {l[0] / float(n[0] - 2),
                                l[1] / float(n[1] - 2),
                                l[2] / float(n[2] - 2)};
    
    SP.setvalue(Piro::SolveParams::DELTA, delta);
    SP.setvalue(Piro::SolveParams::TIMESTEP, std::stof(reader.get("Solve", "Timestep", "default_value")));
    SP.setvalue(Piro::SolveParams::TOTALTIME, std::stof(reader.get("Solve", "TotalTime", "default_value")));
    SP.setvalue(Piro::SolveParams::SAVE, std::stoi(reader.get("Solve", "Save", "default_value")));
    SP.setvalue(Piro::SolveParams::DATATYPE, std::stoi(reader.get("Solve", "Data", "default_value")));
    SP.setvalue(Piro::SolveParams::TOTALTIMESTEPS, static_cast<int>(std::ceil(SP.getvalue<float>(Piro::SolveParams::TOTALTIME) / SP.getvalue<float>(Piro::SolveParams::TIMESTEP))));
    
    if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 2){
        Piro::file_utilities::IniReader reader_data(current_path.string() + "/assets/Data/hashtable.ini");
        SP.setvalue(Piro::SolveParams::PROBING, std::stoi(reader_data.get("Table", "Probing", "default_value")));
        SP.setvalue(Piro::SolveParams::LOADFACTOR, std::stof(reader_data.get("Table", "LoadFactor", "default_value")));
        SP.setvalue(Piro::SolveParams::A, std::stof(reader_data.get("Table", "a", "default_value")));
        SP.setvalue(Piro::SolveParams::B, std::stof(reader_data.get("Table", "b", "default_value")));
        SP.setvalue(Piro::SolveParams::C, std::stof(reader_data.get("Table", "c", "default_value")));
    }
    Piro::bc::readbc();
    Piro::logger::info("Preprocess step completed");
    return 0;
}

bool isBoundaryPoint(int x, int y, int z) {
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
    // Check if the point is on the boundary of any axis
    if (x == 0 || x == n[0] - 1 || y == 0 || y == n[1] - 1 || z == 0 || z == n[2] - 1) {
        return true;
    }
    return false;
}

bool isValidIndex(int index){
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    return (index >= 0 && index < n[0] * n[1] * n[2]);
    }

int Piro::laplacian_CSR_init(){
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<float> l = MP.getvalue<std::vector<float>>(Piro::MeshParams::L);
    int N = n[0] * n[1] * n[2];
    Piro::CellData CD;
    std::vector<CLBuffer> laplacian_collect;
    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD.push_back(CD);
    laplacian_collect.push_back(CD_GPU);
    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].type = 2; // row pointers
    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rowpointers.assign(N + 1, 0.0);
    laplacian_collect.push_back(CD_GPU);
    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].type = 3; // columns
    laplacian_collect.push_back(CD_GPU);
    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].type = 4; // values
    float norm = pow((l[0] / float(n[0] - 2)), 2);
    // Iterate over all grid points
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                
                // Self connection (central point)
                MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(i);
                MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(-6.0/norm);
                
                // Neighbors in the x-direction (x±1)
                if (x > 0) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x-1, y, z, n[0], n[1]));
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                    
                }
                if (x < n[0] - 1) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x+1, y, z, n[0], n[1]));
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                
                }

                // Neighbors in the y-direction (y±1)
                if (y > 0) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x, y-1, z, n[0], n[1]));
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                    
                }
                    if (y < n[1] - 1) {
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x, y+1, z, n[0], n[1]));
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                        
                    }

                    // Neighbors in the z-direction (z±1)
                    if (z > 0) {
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x, y, z-1, n[0], n[1]));
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                        
                    }
                    if (z < n[2] - 1) {
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x, y, z+1, n[0], n[1]));
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                    }
                
                    if (isBoundaryPoint(x, y, z)) {
                        auto it = std::find(MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.begin(),
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.end(), i);
        
                        if (it == MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.end()) {
                            // Add the boundary point if not already in the matrix.
                            MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(i); // for diagonal entry
                            MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(10.0); // large value to enforce the BC
                        }
                        else{
                            size_t index = std::distance(MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.begin(), it);
                            MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values[index] = 10.0;
                        }
                    }

                MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rowpointers[i + 1] = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.size();
            }
        }
    }
    
    cl_int err;
    laplacian_collect[0].buffer = clCreateBuffer(Piro::kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rowpointers.size(), 
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rowpointers.data(), &err);

    laplacian_collect[1].buffer = clCreateBuffer(Piro::kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) *  MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.size(), 
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.data(), &err);

    laplacian_collect[2].buffer = clCreateBuffer(Piro::kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) *  MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.size(), 
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.data(), &err);
    
    LAP_INIT = true;
    CDGPU.setvalue(Piro::CellDataGPU::LAPLACIAN_CSR, laplacian_collect);
    RHS.sparsecount += MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.size();
    // printVector(MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rowpointers);
    // printVector(MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns);
    // printVector(MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values);
    // std::cout << MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.size() << std::endl;
    return 0;
}

