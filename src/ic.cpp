#include <vector>
#include <preprocess.hpp>
#include <init.hpp>
#include <ic.hpp>
#include <extras.hpp>
#include <stringutilities.hpp>
#include <cmath>
#include <logger.hpp>
#include <mathoperations.hpp>
#include <fileutilities.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float Piro::calculategaussian(std::vector<float> coord, std::vector<float> mean, std::vector<float> sigma){
    // Compute normalized differences
    float x = (coord[0] - mean[0]) / sigma[0];
    float y = (coord[1] - mean[1]) / sigma[1];
    float z = (coord[2] - mean[2]) / sigma[2];

    float exponent = -0.5f * (x*x + y*y + z*z);

    float normalization = 1.0f / (std::pow(2.0f * M_PI, 1.5f) * sigma[0] * sigma[1] * sigma[2]);

    return normalization * std::exp(exponent);
}

float Piro::calculatecoulomb(std::vector<float> coord, std::vector<float> center, float Z, double e, double epsilon_0){
    float r = Piro::math_operations::calc_length_3d(coord, center);
    return -(Z * 1E15 * pow(e, 2)) / (4 * M_PI * epsilon_0 * (r + 1E-10));
}

std::vector<float> Piro::initialcondition(int index){
    Piro::logger::info("Initialisation started");
    std::vector<float> values;
    std::srand(std::time(0));
    std::vector<float> coordinate(3);
    
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<float> MP_l = MP.getvalue<std::vector<float>>(Piro::MeshParams::L);
    Piro::file_utilities::IniReader icreader(Piro::file_utilities::current_path.string() + "/assets/IC/" + "distribution.ini");
    if(MP.getvalue<std::vector<std::string>>(Piro::MeshParams::ICFILES)[index] == "Gaussian"){
        values.assign(n[0] * n[1] * n[2], 0.0);
        Piro::logger::info("Gaussian initialisation");
        float scalefactor = std::stof(icreader.get("Gaussian", "Scalefactor", "default_value"));
        std::vector<float> sigma = Piro::string_utilities::convertStringVectorToFloat(Piro::string_utilities::splitString(icreader.get("Gaussian", "Sigma", "default_value"), ' '));
        std::vector<float> mean = Piro::string_utilities::convertStringVectorToFloat(Piro::string_utilities::splitString(icreader.get("Gaussian", "Median", "default_value"), ' '));
        for (int x = 0; x < n[0]; ++x) {
            for (int y = 0; y < n[1]; ++y) {
                for (int z = 0; z < n[2]; ++z) {
                    int l = Piro::math_operations::idx(x, y, z, n[0], n[1]);
                    coordinate[0] = x * MP_l[0] / (float)n[0];
                    coordinate[1] = y * MP_l[1] / (float)n[1];
                    coordinate[2] = z * MP_l[2] / (float)n[2];
                    values[l] = scalefactor * calculategaussian(coordinate, mean, sigma);
                }
            }
        }
    }
    else if (MP.getvalue<std::vector<std::string>>(Piro::MeshParams::ICFILES)[index] == "Coulomb"){
        values.assign(n[0] * n[1] * n[2], 0.0);
        Piro::logger::info("Coulomb initialisation");
        std::vector<float> center = Piro::string_utilities::convertStringVectorToFloat(Piro::string_utilities::splitString(icreader.get("Coulomb", "center", "default_value"), ' '));
        float Z = std::stof(icreader.get("Coulomb", "Z", "default_value"));
        double e = std::stof(icreader.get("Coulomb", "e", "default_value"));
        double epsilon_0 = std::stof(icreader.get("Coulomb", "epsilon_0", "default_value"));
        for (int x = 0; x < n[0]; ++x) {
            for (int y = 0; y < n[1]; ++y) {
                for (int z = 0; z < n[2]; ++z) {
                    int l = Piro::math_operations::idx(x, y, z, n[0], n[1]);
                    coordinate[0] = x * MP_l[0] / n[0];
                    coordinate[1] = y * MP_l[1] / n[1];
                    coordinate[2] = z * MP_l[2] / n[2];
                    values[l] = calculatecoulomb(coordinate, center, Z, e, epsilon_0);
                }
            }
        }
        
    }
    else if (MP.getvalue<std::vector<std::string>>(Piro::MeshParams::ICFILES)[index] == "UniformVector"){
        Piro::logger::info("Uniform vector initialisation");
        values.assign(n[0] * n[1] * n[2] * 3, 0.0);
        std::vector<float> vectordir = Piro::string_utilities::convertStringVectorToFloat(Piro::string_utilities::splitString(icreader.get("Vector", "Direction", "default_value"), ' '));
        std::vector<float> vecval = Piro::string_utilities::convertStringVectorToFloat(Piro::string_utilities::splitString(icreader.get("Vector", "Value", "default_value"), ' '));
        for(int vec = 0; vec < values.size() / 3; vec++){
            values[vec] = vectordir[0] * vecval[0];
        }
        for(int vec = values.size() / 3; vec < 2 * values.size() / 3; vec++){
            values[vec] = vectordir[1] * vecval[0];
        }
        for(int vec = 2 * values.size() / 3; vec < 3 * values.size() / 3; vec++){
            values[vec] = vectordir[2] * vecval[0];
        }
    }
    Piro::logger::info("Initialisation completed");
    return values;

}