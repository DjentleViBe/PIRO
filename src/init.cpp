#include <iostream>
#include <filesystem>
#include <init.hpp>
#include <extras.hpp>
#include <solve.hpp>
#include <fileutilities.hpp>

std::filesystem::path current_path;

void init(){

    if(SP.restart){
        Piro::FileUtilities::delete_directory(current_path.string() + "/" + SP.casename);
    }
    Piro::FileUtilities::create_directory(current_path.string() + "/" + SP.casename);
    Piro::FileUtilities::create_directory(current_path.string() + "/" + SP.casename + "/mesh");
    Piro::FileUtilities::create_directory(current_path.string() + "/" + SP.casename + "/mesh/level");

}
