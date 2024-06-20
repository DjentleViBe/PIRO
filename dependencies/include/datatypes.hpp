#ifndef datatypes_hpp
#define datatypes_hpp

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

    class MeshParams{
        public:
            int meshtype;
            int n[3];
            int l[3];
    };
};

#endif