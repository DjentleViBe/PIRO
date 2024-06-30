#include <iostream>
#include <vector>
#include "../dependencies/include/preprocess.hpp"
#include "../dependencies/include/inireader.hpp"
#include "../dependencies/include/extras.hpp"
#include "../dependencies/include/bc.hpp"
#include "../dependencies/include/solve.hpp"

void initbc(){
    std::vector<int> BC_type;
    std::vector<std::string> BC_property;
    std::vector<float> BC_value;
    std::vector<std::vector<int>> indices(6, std::vector<int>());

    for(int ind = 0; ind < MP.n[0] * MP.n[1] * MP.n[2]; ind++){
        int kd = ind / (MP.n[1] * MP.n[2]);
        int jd = (ind % (MP.n[0] * MP.n[1])) / MP.n[2];
        int id = ind % MP.n[0];

        if(jd == 0 && (id != 0 && id != MP.n[0] - 1) && (kd != 0 && kd != MP.n[2] - 1)){
            // 3 : XZ plane
            indices[3].push_back(ind);
        }
        else if(jd == MP.n[1] - 1 && (id != 0 && id != MP.n[0] - 1) && (kd != 0 && kd != MP.n[2] - 1)){
            // 2 : XZ plane + breadth
            indices[2].push_back(ind);
        }
        
        if(kd == 0 && (id != 0 && id != MP.n[1] - 1) && (jd != 0 && jd != MP.n[1] - 1)){
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

    // printMatrix(indices);

    IniReader reader("setup.ini");
    BC_type = convertStringVectorToInt(splitString(reader.get("BC", "type", "default_value"), ' '));
    BC_property = splitString(reader.get("BC", "property", "default_value"), ' ');
    BC_value = convertStringVectorToFloat(splitString(reader.get("BC", "values", "default_value"), ' '));
    Giro::Solve GS;
    for (int ind = 0; ind < 6; ind++){
        for(int faces = 0; faces < indices[ind].size(); faces++){
            int msv = GS.matchscalartovar(BC_property[ind]);
            //std::cout << indices[ind][faces] << " ";
            MP.AMR[0].CD[msv].values[indices[ind][faces]] = BC_value[ind];
        }
        //std::cout << std::endl;
    }
}

void readbc(){
    std::cout << "Read boundary conditions input" << std::endl;
    IniReader reader("setup.ini");

    if(countSpaces(reader.get("BC", "type", "default_value")) > 1){
        initbc();
    }

}