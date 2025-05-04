#include <iostream>
#include <filesystem>
#include "../dependencies/include/init.hpp"
#include "../dependencies/include/extras.hpp"
#include "../dependencies/include/solve.hpp"
#include "../dependencies/include/fileutilities.hpp"

std::filesystem::path current_path;

void init(){

    if(SP.restart){
        Piro::FileUtilities::delete_directory(current_path.string() + "/" + SP.casename);
    }
    Piro::FileUtilities::create_directory(current_path.string() + "/" + SP.casename);
    Piro::FileUtilities::create_directory(current_path.string() + "/" + SP.casename + "/mesh");
    Piro::FileUtilities::create_directory(current_path.string() + "/" + SP.casename + "/mesh/level");

}
