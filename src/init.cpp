#include <iostream>
#include <filesystem>
#include <init.hpp>
#include <extras.hpp>
#include <solve.hpp>
#include <fileutilities.hpp>

std::filesystem::path current_path;

void init(){

    if(SP.restart){
        Piro::file_utilities::delete_directory(current_path.string() + "/" + SP.casename);
    }
    Piro::file_utilities::create_directory(current_path.string() + "/" + SP.casename);
    Piro::file_utilities::create_directory(current_path.string() + "/" + SP.casename + "/mesh");
    Piro::file_utilities::create_directory(current_path.string() + "/" + SP.casename + "/mesh/level");

}
