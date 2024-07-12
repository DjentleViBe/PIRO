#include <vector>
#include "../dependencies/include/preprocess.hpp"
#include "../dependencies/include/init.hpp"
#include "../dependencies/include/ic.hpp"
#include "../dependencies/include/inireader.hpp"
#include "../dependencies/include/extras.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float calculategaussian(std::vector<float> coord, std::vector<float> mean, std::vector<float> sigma){
    float gaussian = exp(-0.5 * (pow((coord[0] - mean[0])/sigma[0], 2) + 
                                pow((coord[1] - mean[1])/sigma[1], 2) + 
                                pow((coord[2] - mean[2])/sigma[2], 2))) * 1 / 
                                (pow(2 * M_PI, 1.5) * sigma[0] * sigma[1] * sigma[2]);
    
    return gaussian;
}

std::vector<float> initialcondition(int index, int valuetype, int ictype){
    std::vector<float> values;

    if(valuetype == 0){
        // initialise scalar
        switch(ictype){
            case 0:
                values.assign(MP.n[0] * MP.n[1] * MP.n[2], 0.0);
                break;
            case 1:
                values.assign(MP.n[0] * MP.n[1] * MP.n[2], 0.0);
                IniReader icreader(current_path.string() + "/assets/IC/" + MP.ICfiles[index] + ".ini");
                float scalefactor = std::stof(icreader.get("Distribution", "Scalefactor", "default_value"));
                std::vector<float> sigma = convertStringVectorToFloat(splitString(icreader.get("Distribution", "Sigma", "default_value"), ' '));
                std::vector<float> mean = convertStringVectorToFloat(splitString(icreader.get("Distribution", "Median", "default_value"), ' '));

                std::srand(std::time(0));
                std::vector<float> coordinate(3);
                for (int x = 0; x < MP.n[0]; ++x) {
                    for (int y = 0; y < MP.n[1]; ++y) {
                        for (int z = 0; z < MP.n[2]; ++z) {
                            int l = idx(x, y, z, MP.n[0], MP.n[1]);
                            coordinate[0] = x * MP.l[0] / MP.n[0];
                            coordinate[1] = y * MP.l[1] / MP.n[1];
                            coordinate[2] = z * MP.l[2] / MP.n[2];
                            values[l] = scalefactor * calculategaussian(coordinate, mean, sigma);
                        }
                    }
                }
                break;

        }

    }
    else{
        // initialise vector
        switch(ictype){
            case 0:
                values.assign(MP.n[0] * MP.n[1] * MP.n[2] * 3, 0.0);
                break;
            case 1:
                values.assign(MP.n[0] * MP.n[1] * MP.n[2] * 3, 0.0);
                IniReader icreader(current_path.string() + "/assets/IC/" + MP.ICfiles[index] + ".ini");
                std::vector<float> vectordir = convertStringVectorToFloat(splitString(icreader.get("Vector", "Direction", "default_value"), ' '));
                for(int vec = 0; vec < values.size() / 3; vec++){
                    values[vec] = vectordir[0];
                }
                for(int vec = values.size() / 3; vec < 2 * values.size() / 3; vec++){
                    values[vec] = vectordir[1];
                }
                for(int vec = 2 * values.size() / 3; vec < 3 * values.size() / 3; vec++){
                    values[vec] = vectordir[2];
                }
                break;
        }
    }
   
    return values;

}