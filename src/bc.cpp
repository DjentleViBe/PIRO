#include <iostream>
#include <vector>
#include "../dependencies/include/preprocess.hpp"
#include "../dependencies/include/init.hpp"
#include "../dependencies/include/inireader.hpp"
#include "../dependencies/include/extras.hpp"
#include "../dependencies/include/bc.hpp"
#include "../dependencies/include/solve.hpp"
#include <algorithm>

std::vector<std::vector<int>> indices(6, std::vector<int>());
std::vector<int> indices_toprint;
std::vector<int> indices_toprint_vec;
std::vector<std::string> BC_property;
std::vector<float> BC_value;
Giro::Solve GS;

void setbc(){
    for (int ind = 0; ind < 6; ind++){
        for(uint faces = 0; faces < indices[ind].size(); faces++){
            int msv = GS.matchscalartovar(BC_property[ind]);
            MP.AMR[0].CD[msv].values[indices[ind][faces]] = BC_value[ind];
        }
    }
}

void prepbc(){
    std::cout << "Preparing cells to print" << std::endl;
    for(uint ind = 0; ind < MP.n[0] * MP.n[1] * MP.n[2]; ind++){
        uint kd = ind / (MP.n[1] * MP.n[0]);
        uint jd = (ind / MP.n[0]) % MP.n[1];
        uint id = ind % MP.n[0];
        if (id == 0 || id == MP.n[0] - 1 || jd == 0 || jd == MP.n[1] - 1 || kd == 0 || kd == MP.n[2] - 1){
            indices_toprint.push_back(ind);
            indices_toprint_vec.push_back(ind);
            indices_toprint_vec.push_back(ind + MP.n[0]*MP.n[1]*MP.n[2]);
            indices_toprint_vec.push_back(ind + 2 * MP.n[0]*MP.n[1]*MP.n[2]);
            
        }
    }
    std::sort(indices_toprint.rbegin(), indices_toprint.rend());
    std::sort(indices_toprint_vec.rbegin(), indices_toprint_vec.rend());
}

void initbc(){
    std::cout << "Initialising boundary conditions" << std::endl;
    std::vector<int> BC_type;
    
    for(uint ind = 0; ind < MP.n[0] * MP.n[1] * MP.n[2]; ind++){
        uint kd = ind / (MP.n[1] * MP.n[0]);
        uint jd = (ind % (MP.n[1] * MP.n[0])) / MP.n[0];
        uint id = ind % MP.n[0];
        
        if(jd == 0 && (id != 0 && id != MP.n[0] - 1) && (kd != 0 && kd != MP.n[2] - 1)){
            // 3 : XZ plane
            indices[3].push_back(ind);
        }
        else if(jd == MP.n[1] - 1 && (id != 0 && id != MP.n[0] - 1) && (kd != 0 && kd != MP.n[2] - 1)){
            // 2 : XZ plane + breadth
            indices[2].push_back(ind);
        }
        
        if(kd == 0 && (id != 0 && id != MP.n[0] - 1) && (jd != 0 && jd != MP.n[1] - 1)){
            // 4 : XY plane 
            indices[4].push_back(ind);
        }
        else if(kd == MP.n[2] - 1 && (id != 0 && id != MP.n[0] - 1) && (jd != 0 && jd != MP.n[1] - 1)){
            // 5 : XY plane + height
            indices[5].push_back(ind);
        }

        if(id == 0 && (jd != 0 && jd != MP.n[1] - 1) && (kd != 0 && kd != MP.n[2] - 1)){
            // 1 : YZ plane
            indices[1].push_back(ind);
        }
        else if(id == MP.n[0] - 1 && (jd != 0 && jd != MP.n[1] - 1) && (kd != 0 && kd != MP.n[2] - 1)){
            // 0 : YZ plane + length
            indices[0].push_back(ind);
        }
    }

    IniReader reader(current_path.string() + "/assets/setup.ini");
    BC_type = convertStringVectorToInt(splitString(reader.get("BC", "type", "default_value"), ' '));
    BC_property = splitString(reader.get("BC", "property", "default_value"), ' ');
    BC_value = convertStringVectorToFloat(splitString(reader.get("BC", "values", "default_value"), ' '));
    setbc();
    prepbc();
    std::cout << "Boundary conditions initialised" << std::endl;
}

void readbc(){
    std::cout << "Reading boundary conditions" << std::endl;
    IniReader reader(current_path.string() + "/assets/setup.ini");

    if(countSpaces(reader.get("BC", "type", "default_value")) > 1){
        initbc();
    }

}
