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

std::string Piro::paraview::writevti(Piro::AMR AMR){
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::bc::indices& indval = Piro::bc::indices::getInstance();
    std::string level = "";
    level = "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    level += "<ImageData WholeExtent=\""+
                std::to_string(AMR.WholeExtent[0]) + " " +
                std::to_string(AMR.WholeExtent[1] - 2) + " " +
                std::to_string(AMR.WholeExtent[2]) + " " +
                std::to_string(AMR.WholeExtent[3] - 2) + " " +
                std::to_string(AMR.WholeExtent[4]) + " " +
                std::to_string(AMR.WholeExtent[5] - 2) +
                "\" Origin=\"" +
                std::to_string(AMR.Origin[0])+ " " +
                std::to_string(AMR.Origin[1])+ " " +
                std::to_string(AMR.Origin[2]) +
                "\" Spacing=\"" + 
                std::to_string(AMR.Spacing[0])+ " " +
                std::to_string(AMR.Spacing[1])+ " " +
                std::to_string(AMR.Spacing[2]) +
                "\">\n";
    
    level += "<Piece Extent=\"" + 
                std::to_string(AMR.WholeExtent[0]) + " " +
                std::to_string(AMR.WholeExtent[1] - 2) + " " +
                std::to_string(AMR.WholeExtent[2]) + " " +
                std::to_string(AMR.WholeExtent[3] - 2) + " " +
                std::to_string(AMR.WholeExtent[4]) + " " +
                std::to_string(AMR.WholeExtent[5] - 2) +
                "\">\n";
    level += "<PointData Scalars=\"ScalarValues\">\n";
    level += "</PointData>\n";
    level += "<CellData>\n";
    
    for(int l = 0; l < AMR.CD.size() - 3; l++){
        
        if(AMR.CD[l].type == 1){
            
            for (int pos : indval.getvalue<std::vector<int>>(Piro::bc::indices::INDTOPRINT)) {
                if (pos >= 0 && pos < AMR.CD[l].values.size()) {
                    AMR.CD[l].values.erase(AMR.CD[l].values.begin() + pos);
                    }
                }
            level.append("<DataArray type=\"Float64\" Name=\"");
            level.append(AMR.CD[l].Scalars);
            level.append("\" format=\"ascii\">\n");
            level.append(Piro::string_utilities::concatenateStrings2(Piro::string_utilities::floatScalarToString(AMR.CD[l].values)));
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
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::string ts_string = std::to_string(timestep);
    ts_string = std::string(5 - ts_string.length(), '0') + ts_string;
    int offset = 0; 
    std::string vtkfile = "<?xml version=\"1.0\"?>\n";
    vtkfile += "<VTKFile type=\"vtkOverlappingAMR\" version=\"1.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n";
    vtkfile += "<vtkOverlappingAMR origin=\"0 0 0\" grid_description=\"XYZ\">\n";
    for(int i = 0; i < MP.getvalue<int>(Piro::MeshParams::LEVELS); i++){
        int num_blocks = MP.getvalue<std::vector<int>>(Piro::MeshParams::INDEX)[i];
        std::vector<Piro::AMR> amr = MP.getvalue<std::vector<AMR_LEVELS>>(Piro::MeshParams::AMRLEVELS)[i].amr;
        Piro::logger::info(amr.size());
        vtkfile += "<Block level=\"" + std::to_string(i) + "\" spacing=\"1 1 1\">\n";
        for(int j = 0; j < num_blocks; j++){
            vtkfile += "<DataSet index=\"" + std::to_string(j) + "\" ";
            vtkfile += "amr_box=\"" +   std::to_string(amr[j].WholeExtent[0]) + " " +
                                        std::to_string(amr[j].WholeExtent[1] - 2) + " " +
                                        std::to_string(amr[j].WholeExtent[2]) + " " +
                                        std::to_string(amr[j].WholeExtent[3] - 2) + " " +
                                        std::to_string(amr[j].WholeExtent[4]) + " " +
                                        std::to_string(amr[j].WholeExtent[5] - 2) + "\" ";
            vtkfile += "file=\"level/" + ts_string + "_level_" + std::to_string(offset + j) + ".vti\">\n";
            vtkfile += "</DataSet>\n";
            // Write .vti file for the corresponding AMR block
            const auto& amr_block = amr[j];
            std::string filename = Piro::file_utilities::current_path.string() + "/" +
                               SP.getvalue<std::string>(Piro::SolveParams::CASENAME) +
                               "/mesh/level/" + ts_string + "_level_" + std::to_string(offset + j) + ".vti";
            Piro::logger::info("Writing block");
            Piro::file_utilities::writefile(filename, writevti(amr_block));
        }
        vtkfile += "</Block>\n";
        offset += num_blocks;
    }
    vtkfile += "</vtkOverlappingAMR>\n</VTKFile>\n";
    Piro::file_utilities::writefile(Piro::file_utilities::current_path.string() + "/" + SP.getvalue<std::string>(Piro::SolveParams::CASENAME) + "/mesh/mesh_" + ts_string + ".vth", vtkfile);

}
