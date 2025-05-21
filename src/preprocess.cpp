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

int Piro::preprocess(const std::string& name) {
    Piro::CLBuffer CD_GPU;
    Piro::logger::info("Preprocess step initiated");
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    Piro::DeviceParams& DP = Piro::DeviceParams::getInstance();
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::logger& logg = Piro::logger::getInstance();
    Piro::kernels& kernels = Piro::kernels::getInstance();

    Piro::logger::info("Setup file : ",name);
    Piro::file_utilities::IniReader reader(Piro::file_utilities::current_path.string() + "/assets/" + name);
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
    if(kernels.getvalue<int>(Piro::kernels::COMPILE) == 0){
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
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[i].values = initialcondition(i);
        // push scalar data to gpu
        CDGPU_collect.push_back(CD_GPU);
        CDGPU_collect[i].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[i].values.data(), &err);
        j += 1;
    }
    for (int i = j; i < j + MP.getvalue<int>(Piro::MeshParams::VECTORNUM); i++){
        
        CD.Scalars = MP.getvalue<std::vector<std::string>>(Piro::MeshParams::VECTORLIST)[i - j];
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD.push_back(CD);
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[i].type = 1;
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[i].values = initialcondition(i);
        // push vector data to gpu
        CDGPU_collect.push_back(CD_GPU);
        CDGPU_collect[i].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N * 3, MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[i].values.data(), &err);
        
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
        Piro::file_utilities::IniReader reader_data(Piro::file_utilities::current_path.string() + "/assets/Data/hashtable.ini");
        SP.setvalue(Piro::SolveParams::PROBING, std::stoi(reader_data.get("Table", "Probing", "default_value")));
        SP.setvalue(Piro::SolveParams::LOADFACTOR, std::stof(reader_data.get("Table", "LoadFactor", "default_value")));
        SP.setvalue(Piro::SolveParams::A, std::stof(reader_data.get("Table", "a", "default_value")));
        SP.setvalue(Piro::SolveParams::B, std::stof(reader_data.get("Table", "b", "default_value")));
        SP.setvalue(Piro::SolveParams::C, std::stof(reader_data.get("Table", "c", "default_value")));
    }
    Piro::bc::readbc();
    // laplacian operator
    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD.push_back(CD);
    // gradient operator
    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD.push_back(CD);
    // divergence operator
    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD.push_back(CD);
    Piro::logger::info("Preprocess step completed");
    return 0;
}

bool isValidIndex(int index){
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    return (index >= 0 && index < n[0] * n[1] * n[2]);
    }
