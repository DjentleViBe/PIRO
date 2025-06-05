#include <postprocess.hpp>
#include <bc.hpp>
#include <preprocess.hpp>
#include <datatypes.hpp>
#include <init.hpp>
#include <fileutilities.hpp>
#include <stringutilities.hpp>
#include <string>
#include <iostream>
#include <algorithm>
#include <logger.hpp>
#include <octree.hpp>
#include <sstream> 
#include <iomanip>

std::string Piro::paraview::writevti(Piro::AMR AMR, int start, int end, 
                                    std::vector<int> WholeExtent,
                                    std::string Spacing,
                                    std::vector<float> Origin, int lev){
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::bc::indices& indval = Piro::bc::indices::getInstance();
    std::string level = "";
    level = "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    level += "<ImageData WholeExtent=\""+
                std::to_string(WholeExtent[0]) + " " +
                std::to_string(WholeExtent[1]) + " " +
                std::to_string(WholeExtent[2]) + " " +
                std::to_string(WholeExtent[3]) + " " +
                std::to_string(WholeExtent[4]) + " " +
                std::to_string(WholeExtent[5]) +
                "\" Origin=\"" +
                std::to_string(Origin[0])+ " " +
                std::to_string(Origin[1])+ " " +
                std::to_string(Origin[2]) +
                "\" Spacing=\"" + 
                Spacing +
                "\">\n";
    
    level += "<Piece Extent=\"" + 
                std::to_string(WholeExtent[0]) + " " +
                std::to_string(WholeExtent[1]) + " " +
                std::to_string(WholeExtent[2]) + " " +
                std::to_string(WholeExtent[3]) + " " +
                std::to_string(WholeExtent[4]) + " " +
                std::to_string(WholeExtent[5]) +
                "\">\n";
    level += "<PointData Scalars=\"ScalarValues\">\n";
    level += "</PointData>\n";
    level += "<CellData>\n";
    
    for(int l = 0; l < AMR.CD.size() - 3; l++){
        
        if(AMR.CD[l].type == 1){
            if(lev == 0){
                for (int pos : indval.getvalue<std::vector<int>>(Piro::bc::indices::INDTOPRINT)) {
                    if (pos >= 0 && pos < end) {
                        AMR.CD[l].values.erase(AMR.CD[l].values.begin() + pos);
                    }
                }
            }
            level.append("<DataArray type=\"Float64\" Name=\"");
            level.append(AMR.CD[l].Scalars);
            level.append("\" format=\"ascii\">\n");
            level.append(Piro::string_utilities::concatenateStrings2(Piro::string_utilities::floatScalarToStringRange(AMR.CD[l].values, start, end, 2)));
            level.append("</DataArray>\n");
        }
        else if(AMR.CD[l].type == 2){
            for (int pos : indval.getvalue<std::vector<int>>(Piro::bc::indices::INDTOPRINTVEC)) {
                if (pos >= 0 && pos < AMR.CD[l].values.size()) {
                    AMR.CD[l].values.erase(AMR.CD[l].values.begin() + pos);
                    }
                }
            
            if(MP.getvalue<int>(Piro::MeshParams::VECTORNUM) > 0){
                for(int vec = 0; vec < 3; vec++){
                    switch(vec){
                        case 0: 
                            level.append("<DataArray type=\"Float64\" Name=\"");
                            level.append(AMR.CD[l].Scalars + "_x");
                            level.append("\" format=\"ascii\">\n");
                            level.append(Piro::string_utilities::concatenateStrings2(Piro::string_utilities::floatVectorToString(AMR.CD[l].values, 0)));
                            level.append("</DataArray>\n");
                            break;
                        case 1:
                            level.append("<DataArray type=\"Float64\" Name=\"");
                            level.append(AMR.CD[l].Scalars + "_y");
                            level.append("\" format=\"ascii\">\n");
                            level.append(Piro::string_utilities::concatenateStrings2(Piro::string_utilities::floatVectorToString(AMR.CD[l].values, 1)));
                            level.append("</DataArray>\n");
                            break;
                        case 2:
                            level.append("<DataArray type=\"Float64\" Name=\"");
                            level.append(AMR.CD[l].Scalars + "_z");
                            level.append("\" format=\"ascii\">\n");
                            level.append(Piro::string_utilities::concatenateStrings2(Piro::string_utilities::floatVectorToString(AMR.CD[l].values, 2)));
                            level.append("</DataArray>\n");
                            break;
                    }
                }
            }
        }
    }
    
    level += "</CellData>\n";
    level += "</Piece>\n";
    level += "</ImageData>\n";
    level += "</VTKFile>\n";

    return level;
}

void Piro::paraview::writevth(int timestep){
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    int totalcells = MP.getvalue<int>(Piro::MeshParams::TOTALCELLS);
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<float> l = MP.getvalue<std::vector<float>>(Piro::MeshParams::L);
    std::vector<std::vector<int>> pm = MP.getvalue<std::vector<std::vector<int>>>(Piro::MeshParams::MESH);
    std::string ts_string = std::to_string(timestep);
    std::string vtkfile = "<VTKFile type=\"vtkOverlappingAMR\" version=\"1.1\" byte_order=\"LittleEndian\">\n";
    vtkfile += "<vtkOverlappingAMR origin=\"0 0 0\" grids=\"";
    vtkfile += std::to_string(totalcells);
    vtkfile += "\" refinementRatios=\"2 2 2\">\n\n";
    // int sub = 0;
    auto& cd = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0];
    std::vector<int> levelcount = MP.getvalue<std::vector<int>>(Piro::MeshParams::LEVELCOUNT);
    int start = 0;
    int end = 0;
    std::vector<int> WE(6);
    std::vector<float> OR(3);
    std::vector<float> Spacing(3);

    for(int i = 0; i < MP.getvalue<int>(Piro::MeshParams::LEVELS); i++){
        float spacing_x = l[0] / (float)(n[0]) / std::pow(2, i);
        float spacing_y = l[1] / (float)(n[1]) / std::pow(2, i);
        float spacing_z = l[2] / (float)(n[2]) / std::pow(2, i);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << spacing_x << " " << spacing_y << " "<< spacing_z;
        std::string spacingStr = oss.str();
        Piro::logger::info(spacingStr);
        vtkfile += "<Block level=\"";
        vtkfile += std::to_string(i);
        vtkfile += "\" spacing=\"";
        vtkfile += spacingStr;
        vtkfile += "\">\n";
        int locind = 0;
        for(int lev = 0; lev < pm.size(); lev++){
            if(i == 0){
                // sub = 0;
                WE = {0, 1, 0, 1, 0, 1};
                end = start + 1;
            }
            else{
                // sub = 0;
                WE = {0, 2, 0, 2, 0, 2};
                end = start + (8 * (pm[lev][3] - pm[lev][2]) * (pm[lev][5] - pm[lev][4]) * (pm[lev][7] - pm[lev][6]));
            }
            if(pm[lev][0] == i){
                
                if(i == 0){
                    vtkfile += "<DataSet index=\"" + std::to_string(locind) + "\" ";
                    vtkfile += "amr_box=\"" + 
                            std::to_string(pm[lev][8] % n[0]) + " " +
                            std::to_string(pm[lev][8] % n[0]) + " " +
                            std::to_string((pm[lev][8] / n[0]) % n[1]) + " " +
                            std::to_string((pm[lev][8] / n[0]) % n[1]) + " " +
                            std::to_string(pm[lev][8] / (n[0] * n[1])) + " " +
                            std::to_string(pm[lev][8] / (n[0] * n[1])) + "\" ";
                }
                else{
                    vtkfile += "<DataSet index=\"" + std::to_string(pm[lev][1]) + "\" ";
                    vtkfile += "amr_box=\"" + 
                                std::to_string(pm[lev][2]) + " " +
                                std::to_string(pm[lev][3]) + " " +
                                std::to_string(pm[lev][4]) + " " +
                                std::to_string(pm[lev][5]) + " " +
                                std::to_string(pm[lev][6]) + " " +
                                std::to_string(pm[lev][7]) + "\" ";
                }
                vtkfile += "file=\"level/" + ts_string + "_level_" + std::to_string(i) + "_" + std::to_string(locind) + ".vti\" />\n";
                

                // write vti file here ---->
                std::string filename = Piro::file_utilities::current_path.string() + "/" +
                                        SP.getvalue<std::string>(Piro::SolveParams::CASENAME) +
                                        "/mesh/level/" + ts_string + "_level_" + std::to_string(i) 
                                        + "_" + std::to_string(locind) + ".vti";
                // calculate the origin
                OR = Piro::mesh_operations::octree::get_origin(pm[lev]);
                Piro::file_utilities::writefile(filename, writevti(cd, start, end, WE, spacingStr, OR, i));
                locind++;
                start = end;
            }
        }
        vtkfile += "</Block>\n\n";
        
    }
    vtkfile += "</vtkOverlappingAMR>\n</VTKFile>";
    Piro::file_utilities::writefile(Piro::file_utilities::current_path.string() + "/" + 
                                    SP.getvalue<std::string>(Piro::SolveParams::CASENAME) + 
                                    "/mesh/mesh_" + ts_string + ".vth", vtkfile);

}
