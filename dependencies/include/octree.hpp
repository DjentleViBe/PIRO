#ifndef octree_hpp
#define octree_hpp
#include <vector>

namespace Piro::mesh_operations::octree{
    std::vector<std::vector<float>> get_origin(std::vector<int> ind);
    int findind(std::vector<std::vector<int>> pm, int match, int level);
    std::vector<float> get_ijk(std::vector<int> ind, std::vector<float>& coord, std::vector<std::vector<int>> pm);
}
#endif