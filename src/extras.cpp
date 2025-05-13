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
}
