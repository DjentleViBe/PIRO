#include <iostream>
#include <filesystem>
#include "../dependencies/include/init.hpp"
#include "../dependencies/include/extras.hpp"
#include "../dependencies/include/solve.hpp"
#include "../dependencies/include/fileutilities.hpp"

std::filesystem::path current_path;

void init(){

    if(SP.restart){
        delete_directory(current_path.string() + "/" + SP.casename);
    }
    create_directory(current_path.string() + "/" + SP.casename);
    create_directory(current_path.string() + "/" + SP.casename + "/mesh");
    create_directory(current_path.string() + "/" + SP.casename + "/mesh/level");

}
