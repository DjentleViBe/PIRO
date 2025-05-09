#ifndef datatypes_hpp
#define datatypes_hpp
#include <vector>
#include <string>
#include <unordered_map>
#include <variant>
#include <iostream>

typedef unsigned int uint;

namespace Piro{
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
            std::vector<int> columns;
            std::vector<int> rows;
            std::vector<int> rowpointers;
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
            static MeshParams& getInstance(){
                static MeshParams instance;
                return instance;
            }
            MeshParams(const MeshParams&) = delete;
            MeshParams& operator=(const MeshParams&) = delete;
            enum ParameterIndex{
                // int
                num_cells, MESHTYPE, LEVELS, SCALARNUM, VECTORNUM, INDEX, ICTYPE,
                // float
                O, S, L, CONSTANTSVALUES,
                // string
                CONSTANTSLIST, SCALARLIST, VECTORLIST, ICFILES,
                // custom
                AMR
            };


            template<typename T>
            void setvalue(const ParameterIndex paramindex, const T& val) {
                parameters[paramindex] = val;
            }
            // Specialization for arrays (e.g., std::vector)
            template<typename T>
            void setvalue(const ParameterIndex paramindex, const std::vector<T>& val) {
                parameters[paramindex] = val;
            }

            template<typename T>
            T& getvalue(const ParameterIndex paramindex) {
                if (parameters.find(paramindex) == parameters.end()) {
                    throw std::runtime_error("Parameter not set");
                }
                if (!std::holds_alternative<T>(parameters.at(paramindex))) {
                    std::cout << "bad variant : " << paramindex;
                    throw std::bad_variant_access(); // Throwing a specific exception for mismatched types
                }
            return std::get<T>(parameters.at(paramindex));
            }
            /*
            std::vector<AMR> AMR;*/
        
        private:
            MeshParams() = default;
            using ParamValue = std::variant<int, uint, float, std::string, std::vector<int>, std::vector<uint>, std::vector<float>, std::vector<std::string>, std::vector<Piro::AMR>>;
            std::unordered_map<ParameterIndex, ParamValue> parameters;
    };

    class SolveParams{
        public:
            static SolveParams& getInstance(){
                static SolveParams instance;
                return instance;
            }

            SolveParams(const SolveParams&) = delete;
            SolveParams& operator=(const SolveParams&) = delete;
            enum ParameterIndex{
                // int 
                SOLVERSCHEME, TIMESCHEME, SPACESCHEME, RESTART, TOTALTIMESTEPS,
                SAVE, DATATYPE, PROBING, 
                // float
                A, B, C, TIMESTEP, TOTALTIME, DELTA, DELTAT, LOADFACTOR,
                // string
                CASENAME
            };
            template<typename T>
            void setvalue(const ParameterIndex paramindex, const T& val) {
                parameters[paramindex] = val;
            }
            // Specialization for arrays (e.g., std::vector)
            template<typename T>
            void setvalue(const ParameterIndex paramindex, const std::vector<T>& val) {
                parameters[paramindex] = val;
            }

            template<typename T>
            T getvalue(const ParameterIndex paramindex) const {
                if (parameters.find(paramindex) == parameters.end()) {
                    throw std::runtime_error("Parameter not set");
                }
                if (!std::holds_alternative<T>(parameters.at(paramindex))) {
                    std::cout << "bad variant : " << paramindex;
                    throw std::bad_variant_access(); // Throwing a specific exception for mismatched types
                }
                return std::get<T>(parameters.at(paramindex));
            }

        private:
            SolveParams() = default;
            using ParamValue = std::variant<int, float, std::string, std::vector<int>, std::vector<float>, std::vector<std::string>>;
            std::unordered_map<ParameterIndex, ParamValue> parameters;
        
    };

    class DeviceParams{
        public:
            static DeviceParams& getInstance(){
                static DeviceParams instance;
                return instance;
            }

            DeviceParams(const DeviceParams&) = delete;
            DeviceParams& operator=(const DeviceParams&) = delete;
            enum ParameterIndex{
                // int
                ID, PLATFORMID, TYPE
            };
            template<typename T>
            void setvalue(const ParameterIndex paramindex, const T& val) {
                parameters[paramindex] = val;
            }

            template<typename T>
            T getvalue(const ParameterIndex paramindex) const {
                if (parameters.find(paramindex) == parameters.end()) {
                    throw std::runtime_error("Parameter not set");
                }
                return std::get<T>(parameters.at(paramindex));
            }
        private:
            DeviceParams() = default;
            using ParamValue = std::variant<int, double, std::string>;
            std::unordered_map<ParameterIndex, ParamValue> parameters;
        
    };
};

#endif