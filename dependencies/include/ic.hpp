#ifndef ic_hpp
#define ic_hpp

#include <vector>

namespace Piro{
    /**
     * @brief Initialisation.
     *
     * This function calculates values to be assigned to grid cells
     * based on initialisation type based on /assets/IC/distribution.ini.
     *
     * @param index type of initialisation function. 
     * @return Vector of size n_x * n_y * n_z containing the calcalated values.
     */
    std::vector<float> initialcondition(int index);
    /**
     * @brief Gaussian function.
     *
     * This function calculates grid alues based on gaussian distribution.
     *
     * @param coord cell @ x, y, z. 
     * @param mean mean of the gaussian distribution.
     * @param sigma standard deviation of the gaussian distribution.
     * @return calculated value.
     */
    float calculategaussian(std::vector<float> coord, std::vector<float> mean, std::vector<float> sigma);
    /**
     * @brief Coulomb function.
     *
     * This function calculates grid alues based on coulomb sphere.
     *
     * @param coord cell @ x, y, z. 
     * @param center center location.
     * @param Z atomic number.
     * @param e Charge value
     * @param epsilon_0 permittivity.
     * @return calculated value.
     */
    float calculatecoulomb(std::vector<float> coord, std::vector<float> center, float Z, double e, double epsilon_0);
}
#endif