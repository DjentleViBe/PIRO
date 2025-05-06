#include <iostream>
#include <filesystem>
#include <init.hpp>
#include <extras.hpp>
#include <process.hpp>
#include <fileutilities.hpp>
#include <logger.hpp>

std::filesystem::path current_path;

void Piro::init(){
    Piro::Logger::info("Directory INIT");
    if(SP.restart){
        Piro::file_utilities::delete_directory(current_path.string() + "/" + SP.casename);
    }
    Piro::file_utilities::create_directory(current_path.string() + "/" + SP.casename);
    Piro::file_utilities::create_directory(current_path.string() + "/" + SP.casename + "/mesh");
    Piro::file_utilities::create_directory(current_path.string() + "/" + SP.casename + "/mesh/level");
    Piro::Logger::info("Directory INIT completed\n");
}
