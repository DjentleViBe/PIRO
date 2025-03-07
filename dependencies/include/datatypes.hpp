#ifndef datatypes_hpp
#define datatypes_hpp
#include <vector>
#include <string>

typedef unsigned int uint;

namespace Giro{
    class fvMatrix{
        public:
            typedef int d1scalar_i;
            typedef int d2vector_i[2];
            typedef int d3vector_i[3];
            typedef double d1scalar_d;
            typedef double d2vector_d[2];
            typedef double d3vector_d[3];

    };

    class PointData{
        public:
            std::string Scalars;
            std::vector<float> values;
    };

    class CellData{
        public:
            int type; // 0 : Scalars, 1 : Vectors, 2 : CSR row pointers, 3 : CSR columns, 4 : CSR values
            std::string Scalars;
            std::vector<float> values;
            std::vector<int> indices;
    };

    class AMR{
        public:
            int WholeExtent[6];
            int PieceExtent[6];
            int Origin[3];
            int Spacing[3];
            std::vector<PointData> PD;
            std::vector<CellData> CD;
    };

    class MeshParams{
        public:
            int meshtype;
            int levels;
            int scalarnum;
            int vectornum;
            std::vector<int> ICtype;
            std::vector<std::string> ICfiles;
            std::string ICfile;
            std::vector<uint> n;
            std::vector<float> o;
            std::vector<float> s;
            std::vector<float> l;
            std::vector<int> index;
            std::vector<std::string> constantslist;
            std::vector<float> constantsvalues;
            std::vector<std::string> scalarlist;
            std::vector<std::string> vectorlist;
            std::vector<AMR> AMR;
    };

    class SolveParams{
        public:
            int timescheme;
            int spacescheme;
            std::string casename;
            int restart;
            float timestep;
            float totaltime;
            float delta[3];
            float deltaT;
            int totaltimesteps;
            int save;
    };
    
    class DeviceParams{
        public:
            int id;
            int platformid;
            int type;
    };
};

#endif