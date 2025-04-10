#ifndef extras_hpp
#define extras_hpp

#include <string>
#include <vector>
#include "datatypes.hpp"

int writefile(std::string file_path, std::string line_to_write);
int create_directory(std::string directoryname);
int delete_directory(std::string folderPath);
int get_exec_directory();
std::vector<std::string> splitString(const std::string& str, char delimiter);
std::vector<int> convertStringVectorToInt(const std::vector<std::string>& stringVector);
std::vector<uint> convertStringVectorToUInt(const std::vector<std::string>& stringVector);
std::vector<float> convertStringVectorToFloat(const std::vector<std::string>& stringVector);
std::vector<std::string> floatVectorToString(const std::vector<float>& floatVector, int dex);
std::vector<std::string> floatScalarToString(const std::vector<float>& floatScalar);
std::string concatenateStrings(const std::vector<std::string>& strVector);
int countSpaces(const std::string& str);
std::string concatenateStrings2(const std::vector<std::string>& strVector);
void printMatrix(const std::vector<std::vector<float>>& matrix);
void printMatrix(const std::vector<std::vector<int>>& matrix);
void printVector(const std::vector<float>& vec);
void printVector(const std::vector<int>& vec);
void printArray(float* array, uint size);
std::vector<int> flattenvector(std::vector<std::vector<int>> twoDVector);
void print_time();
int countWords(const std::string& str);
#endif