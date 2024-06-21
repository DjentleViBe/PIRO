#include <iostream>
#include "../dependencies/include/postprocess.hpp"
#include "../dependencies/include/preprocess.hpp"

int postprocess() {
    std::cout << "This is the postprocess source" << std::endl;
    writevth();
    return 0;
}