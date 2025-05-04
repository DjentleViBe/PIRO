#include <vector>
#include <sstream>
#include <numeric>
#include <iterator>
#include <extras.hpp>
#include <logger.hpp>

namespace Piro::string_utilities{
    std::vector<std::string> splitString(const std::string& str, char delimiter) {
        std::vector<std::string> tokens;
        std::size_t start = 0, end = 0;

        while ((end = str.find(delimiter, start)) != std::string::npos) {
            tokens.push_back(str.substr(start, end - start));
            start = end + 1;
        }

        tokens.push_back(str.substr(start));
        return tokens;
    }

    std::vector<int> convertStringVectorToInt(const std::vector<std::string>& stringVector) {
        std::vector<int> intVector;
        
        for (const std::string& str : stringVector) {
            std::istringstream iss(str);
            int num;
            if (iss >> num) {
                intVector.push_back(num);
            } else {
                // Handle error if conversion fails
                Piro::Logger::info("Error: Conversion failed for string 's", str, "'");
                // You may choose to throw an exception or handle the error in another way
            }
        }
        
        return intVector;
    }

    std::vector<uint> convertStringVectorToUInt(const std::vector<std::string>& stringVector) {
        std::vector<uint> intVector;
        
        for (const std::string& str : stringVector) {
            std::istringstream iss(str);
            uint num;
            if (iss >> num) {
                intVector.push_back(num);
            } else {
                // Handle error if conversion fails
                Piro::Logger::info("Error: Conversion failed for string '", str, "'");
                // You may choose to throw an exception or handle the error in another way
            }
        }
        
        return intVector;
    }

    std::vector<float> convertStringVectorToFloat(const std::vector<std::string>& stringVector) {
        std::vector<float> intVector;
        
        for (const std::string& str : stringVector) {
            std::istringstream iss(str);
            float num;
            if (iss >> num) {
                intVector.push_back(num);
            } else {
                // Handle error if conversion fails
                Piro::Logger::info("Error: Conversion failed for string '", str, "'");
                // You may choose to throw an exception or handle the error in another way
            }
        }
        
        return intVector;
    }

    std::vector<std::string> floatScalarToString(const std::vector<float>& floatScalar) {
        std::vector<std::string> stringScalar;

        // Iterate through each float element and convert to string
        for (float value : floatScalar) {
            // Convert float to string
            std::ostringstream oss;
            oss << value;
            stringScalar.push_back(oss.str());
        }

        return stringScalar;
    }

    std::vector<std::string> floatVectorToString(const std::vector<float>& floatVector, int dex) {
        std::vector<std::string> stringVector;
        // Iterate through each float element and convert to string
        for (int i = dex * floatVector.size() / 3; i < (dex + 1) * floatVector.size() / 3; ++i) {
            // Convert float to string
            std::ostringstream oss;
            oss << floatVector[i];
            stringVector.push_back(oss.str());
        }

        return stringVector;
    }

    std::string concatenateStrings(const std::vector<std::string>& strVector) {
        // Use std::accumulate to concatenate all strings in the vector
        return std::accumulate(strVector.begin(), strVector.end(), std::string(""));
    }

    std::string concatenateStrings2(const std::vector<std::string>& strVector) {
        // Use std::accumulate to concatenate all strings in the vector
    
        return std::accumulate(strVector.begin(), strVector.end(), std::string(""),
                            [](const std::string& acc, const std::string& str) {
                                return acc.empty() ? str : acc + " " + str;
                            });
    }

    int countSpaces(const std::string& str) {
        int spaceCount = 0;

        // Iterate through each character in the string
        for (char ch : str) {
            // Check if the character is a space
            if (ch == ' ') {
                spaceCount++;
            }
        }

        return spaceCount;
    }

    int countWords(const std::string& str) {
        std::istringstream iss(str);
        return std::distance(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>());
    }
}