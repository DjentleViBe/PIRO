#ifndef stringutilities_hpp
#define stringutilities_hpp
#include <vector>
#include <string>

namespace Piro::string_utilities{
    /**
     * @brief Split a string.
     *
     * Split a given string at the delimiter.
     * @param str string to be split
     * @param delimiter delimiter entry
     * @return vector containing parts of the split string
     */
    std::vector<std::string> splitString(const std::string& str, char delimiter);
    /**
     * @brief Convert a string into integer.
     *
     * Given a vector of strings, convert it to integer data type.
     * @param stringVector vector of strings to be converted.
     * @return vector containing the converted data
     */
    std::vector<int> convertStringVectorToInt(const std::vector<std::string>& stringVector);
    /**
     * @brief Convert a string into undigned integer.
     *
     * Given a vector of strings, convert it to undigned integer data type.
     * @param stringVector vector of strings to be converted.
     * @return vector containing the converted data
     */
    std::vector<uint> convertStringVectorToUInt(const std::vector<std::string>& stringVector);
    /**
     * @brief Convert a string into float.
     *
     * Given a vector of strings, convert it to float data type.
     * @param stringVector vector of strings to be converted.
     * @return vector containing the converted data
     */
    std::vector<float> convertStringVectorToFloat(const std::vector<std::string>& stringVector);
    /**
     * @brief Convert a float VECTOR into string.
     *
     * Given a VECTOR of floats , convert it to string data type.
     * @param floatVector VECTOR of floats to be converted.
     * @param dex previous vectro position
     * @return VECTOR containing the converted data
     */
    std::vector<std::string> floatVectorToString(const std::vector<float>& floatVector, int dex);
    /**
     * @brief Convert a float SCALAR into string.
     *
     * Given a SCALAR of floats , convert it to string data type.
     * @param floatScalar SCALAR of floats to be converted.
     * @return SCALAR containing the converted data
     */
    std::vector<std::string> floatScalarToString(const std::vector<float>& floatScalar);
    /**
     * @brief concatenate strings.
     *
     * Join all strings into one big string.
     * @param strVector vector of string.
     * @return combined string 
     */
    std::string concatenateStrings(const std::vector<std::string>& strVector);
    /**
     * @brief concatenate strings.
     *
     * Join all strings into one big string.
     * @param strVector vector of string.
     * @return combined string 
     */
    std::string concatenateStrings2(const std::vector<std::string>& strVector);
    /**
     * @brief count spaces.
     *
     * alculate the number of spaces in a given string.
     * @param str string input.
     * @return number of spaces 
     */
    int countSpaces(const std::string& str);
    /**
     * @brief count words.
     *
     * calculate the number of words in a given string.
     * @param str string input.
     * @return number of words 
     */
    int countWords(const std::string& str);
}

#endif