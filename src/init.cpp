#include <iostream>
#include <filesystem>
#include "../dependencies/include/init.hpp"
#include "../dependencies/include/extras.hpp"
#include "../dependencies/include/solve.hpp"

std::filesystem::path current_path;

void init(){

    get_exec_directory();
    if(SP.restart){
        delete_directory(current_path.string() + "/" + SP.casename);
    }
    create_directory(current_path.string() + "/" + SP.casename);
    create_directory(current_path.string() + "/" + SP.casename + "/mesh");
    create_directory(current_path.string() + "/" + SP.casename + "/mesh/level");

}
