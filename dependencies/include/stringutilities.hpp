#ifndef stringutilities_hpp
#define stringutilities_hpp
#include <vector>
#include <string>

std::vector<std::string> splitString(const std::string& str, char delimiter);
std::vector<int> convertStringVectorToInt(const std::vector<std::string>& stringVector);
std::vector<uint> convertStringVectorToUInt(const std::vector<std::string>& stringVector);
std::vector<float> convertStringVectorToFloat(const std::vector<std::string>& stringVector);
std::vector<std::string> floatVectorToString(const std::vector<float>& floatVector, int dex);
std::vector<std::string> floatScalarToString(const std::vector<float>& floatScalar);
std::string concatenateStrings(const std::vector<std::string>& strVector);
std::string concatenateStrings2(const std::vector<std::string>& strVector);
int countSpaces(const std::string& str);

#endif