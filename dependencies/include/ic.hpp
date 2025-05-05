#ifndef ic_hpp
#define ic_hpp

#include <vector>

namespace Piro{
    std::vector<float> initialcondition(int index, int valuetype);
    float calculategaussian(std::vector<float> coord, std::vector<float> mean, std::vector<float> sigma);
    float calculatecoulomb(std::vector<float> coord, std::vector<float> center, float Z, double e, double epsilon_0);
}
#endif