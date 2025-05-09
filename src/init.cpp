#include <iostream>
#include <filesystem>
#include <init.hpp>
#include <extras.hpp>
#include <process.hpp>
#include <fileutilities.hpp>
#include <logger.hpp>
#include <datatypes.hpp>

std::filesystem::path current_path;

void Piro::init(){
    Piro::logger::info("Directory INIT");
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    std::string casename = SP.getvalue<std::string>(Piro::SolveParams::CASENAME);
    if(SP.getvalue<int>(Piro::SolveParams::RESTART)){
        Piro::file_utilities::delete_directory(current_path.string() + "/" + casename);
    }
    Piro::file_utilities::create_directory(current_path.string() + "/" + casename);
    Piro::file_utilities::create_directory(current_path.string() + "/" + casename + "/mesh");
    Piro::file_utilities::create_directory(current_path.string() + "/" + casename + "/mesh/level");
    Piro::logger::info("Directory INIT completed\n");
}
