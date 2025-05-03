# IC
This section is used to give details on various fields are parameters used in ```.ini``` files for Initial Condition under ```/dependencies/assets/IC```.

## distribution.ini
The scalars and vectors used in the simulation can be initialised with the following predefined functions.

### [Gaussian]
The probability density function of a Gaussian distribution is given by:
$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$
Since the block is in 3D, the ```x, y, z``` values for each parameter in from the equation should be seperated by white space.

### [Vector]
Sets the vector values of each grid.

### [Coulomb]
The coulomb potential is given by:
$$
V(r) = -\frac{Z \cdot e^2 \cdot 10^{15}}{4\pi \varepsilon_0 (r + 10^{-10})}
$$
Since the block is in 3D, the ```x, y, z``` values for ```center``` should be separated by whitespace.