#include <vector>

namespace Piro::vector_operations{
    std::vector<int> flattenvector(std::vector<std::vector<int>> twoDVector){
        std::vector<int> flatVector;
        for (const auto& row : twoDVector) {
            for (int elem : row) {
                flatVector.push_back(elem);
            }
        }
        return flatVector;
    }

    std::vector<float> flattenvector(std::vector<std::vector<float>> twoDVector){
        std::vector<float> flatVector;
        for (const auto& row : twoDVector) {
            for (int elem : row) {
                flatVector.push_back(elem);
            }
        }
        return flatVector;
    }
    
    float sum_numbers(const std::vector<float>& numbers) {
        float sum = 0.0f;
        for (float num : numbers) {
            sum += num * num;
        }
    return sum;
    }
    void subtractvectors(std::vector<float>& number1, const std::vector<float>& number2) {
       for(int i = 0; i < number1.size(); i++){
            number1[i] = number1[i] - number2[i];
       }
    }
}

