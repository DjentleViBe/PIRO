#include "../dependencies/include/postprocess.hpp"
#include "../dependencies/include/bc.hpp"
#include "../dependencies/include/preprocess.hpp"
#include "../dependencies/include/datatypes.hpp"
#include "../dependencies/include/init.hpp"
#include "../dependencies/include/extras.hpp"
#include <string>
#include <iostream>
#include <algorithm>

std::string writevti(Giro::AMR AMR){
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
    
    std::sort(indices_toprint.rbegin(), indices_toprint.rend());

    for(int l = 0; l < AMR.CD.size() - 1; l++){
        for (int pos : indices_toprint) {
            if (pos >= 0 && pos < AMR.CD[l].values.size()) {
                AMR.CD[l].values.erase(AMR.CD[l].values.begin() + pos);
            }
        }
        
        level.append("<DataArray type=\"Float64\" Name=\"");
        level.append(AMR.CD[l].Scalars);
        level.append("\" format=\"ascii\">\n");
        level.append(concatenateStrings2(floatVectorToString(AMR.CD[l].values)));
        level.append("</DataArray>\n");
        
    }
    
    level += "</CellData>\n";
    level += "</Piece>\n";
    level += "</ImageData>\n";
    level += "</VTKFile>\n";

    return level;
}

void writevth(){
    std::string vtkfile = "<?xml version=\"1.0\"?>\n";
    vtkfile += "<VTKFile type=\"vtkOverlappingAMR\" version=\"1.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n";
    vtkfile += "<vtkOverlappingAMR origin=\"0 0 0\" grid_description=\"XYZ\">\n";
    for(int i=0; i < MP.levels; i++){
        for(int j = 0; j < MP.index[i]; j++){
            vtkfile += "<Block level=\""+ std::to_string(i) +"\" spacing=\"1 1 1\">\n";
            vtkfile += "<DataSet index=\"0\" ";
            vtkfile += "amr_box=\"0 " + std::to_string(MP.n[0] - 2) + " " +
                        std::to_string(0) + " " +
                        std::to_string(MP.n[1] - 2) + " " +
                        std::to_string(0) + " " +
                        std::to_string(MP.n[2] - 2) + "\" ";
            vtkfile += "file=\"level/level_" + std::to_string(j) + ".vti\">\n";
            vtkfile += "</DataSet>\n";
            writefile(current_path.string() + "/mesh/level/level_" + std::to_string(j) + ".vti", writevti(MP.AMR[j]));
        }
        vtkfile += "</Block>\n";
    }
    vtkfile += "</vtkOverlappingAMR>\n</VTKFile>\n";
    writefile(current_path.string() + "/mesh/mesh.vth", vtkfile);

}
