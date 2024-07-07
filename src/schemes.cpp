#include "../dependencies/include/preprocess.hpp"

void MA_3(){
    for (int k = 0; k < MP.n[2]; ++k) {
            for (int j = 0; j < MP.n[1]; ++j) {
                for (int i = 0; i < MP.n[0]; ++i) {
                    int l = idx(i, j, k, MP.n[0], MP.n[1]);

                    // Diagonal entry
                    scalapmatrix[l][l] = -2 * SP.timestep * (1/(SP.delta[0] * SP.delta[0]) + 1/(SP.delta[1] * SP.delta[1]) + 1/(SP.delta[2] * SP.delta[2]));
                    scagradmatrix[l][l] = 0;

                    // Off-diagonal entries
                    if (i > 0) {
                        scalapmatrix[l][idx(i-1, j, k, MP.n[0], MP.n[1])] = 1 * SP.timestep / (SP.delta[0] * SP.delta[0]);
                        scagradmatrix[l][idx(i-1, j, k, MP.n[0], MP.n[1])] = -SP.timestep / (2 * SP.delta[0]);
                        
                    }
                    if (i < MP.n[0] - 1) {
                        scalapmatrix[l][idx(i+1, j, k, MP.n[0], MP.n[1])] = 1 * SP.timestep / (SP.delta[0] * SP.delta[0]);
                        scagradmatrix[l][idx(i+1, j, k, MP.n[0], MP.n[1])] = SP.timestep / (2 * SP.delta[0]);
                        
                    }
                    if (j > 0) {
                        scalapmatrix[l][idx(i, j-1, k, MP.n[0], MP.n[1])] = 1 * SP.timestep / (SP.delta[1] * SP.delta[1]);
                        scagradmatrix[l][idx(i, j-1, k, MP.n[0], MP.n[1])] = -SP.timestep / (2 * SP.delta[1]);
                        
                    }
                    if (j < MP.n[1] - 1) {
                        scalapmatrix[l][idx(i, j+1, k, MP.n[0], MP.n[1])] = 1 * SP.timestep / (SP.delta[1] * SP.delta[1]);
                        scagradmatrix[l][idx(i, j+1, k, MP.n[0], MP.n[1])] = SP.timestep / (2 * SP.delta[1]);
                        
                    }
                    if (k > 0) {
                        scalapmatrix[l][idx(i, j, k-1, MP.n[0], MP.n[1])] = 1 * SP.timestep / (SP.delta[2] * SP.delta[2]);
                        scagradmatrix[l][idx(i, j, k-1, MP.n[0], MP.n[1])] = -SP.timestep / (2 * SP.delta[2]);
                        
                    }
                    if (k < MP.n[2] - 1) {
                        scalapmatrix[l][idx(i, j, k+1, MP.n[0], MP.n[1])] = 1 * SP.timestep / (SP.delta[2] * SP.delta[2]);
                        scagradmatrix[l][idx(i, j, k+1, MP.n[0], MP.n[1])] = SP.timestep / (2 * SP.delta[2]);
                        
                    }
                }
            }
        }
}

void schemes(){
    switch(SP.spacescheme){
        case 3:
            MA_3();
            break;

    }
}