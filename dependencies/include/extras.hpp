#ifndef extras_hpp
#define extras_hpp

#include <string>
#include <vector>

int writefile(std::string file_path, std::string line_to_write);
int create_directory(std::string directoryname);
int get_exec_directory();
std::vector<std::string> splitString(const std::string& str, char delimiter);
std::vector<int> convertStringVectorToInt(const std::vector<std::string>& stringVector);
std::vector<float> convertStringVectorToFloat(const std::vector<std::string>& stringVector);
std::vector<std::string> floatVectorToString(const std::vector<float>& floatVector);
std::string concatenateStrings(const std::vector<std::string>& strVector);
int countSpaces(const std::string& str);
std::string concatenateStrings2(const std::vector<std::string>& strVector);
void printMatrix(const std::vector<std::vector<float>>& matrix);
void printMatrix(const std::vector<std::vector<int>>& matrix);
void printVector(const std::vector<float>& vec);
void printVector(const std::vector<int>& vec);

#endif