<p align="center">
 <img width="150px" src="./pics/PIRO_logo_compressed.png" align="center" alt="PIRO Readme" />
 <h1 align="center">PIRO</h1>
 <p align="center" style="font-size: 18px;">G<b>P</b>U / C<b>P</b>U dr<b>I</b>ven nume<b>R</b>ical s<b>O</b>lver</p>


<p align="center">Powered by : 
    <a href="https://www.khronos.org/opencl/">
        <img width="110px" src="https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenCL_logo.svg" align="center" alt="OpenCL"/>
    </a>
</p>

<p align="center">
    <img width="350px" src="./pics/Readme_status.png" align="center" alt="PIRO Readme" />
</p>

<h2> </h2>

## Table of Contents
- [About](#about)
- [Features](#features)
- [Running](#running)
- [How to Build](#how-to-build)
- [Documentation](#documentation)
- [Feedback and Contributions](#feedback-and-contributions)
- [License](#license)

## About 
**PIRO** is an open-source C++/OpenCL software for performing high-performance numerical simulations. It provides modular solvers, supports various sparse formats (CSR, COO), and offers kernel-level customization for advanced users. PIRO is designed for research and production workflows requiring scalable parallel matrix computations.
## Features
- __Cross-platform CPU / GPU operation__ via OpenCL.
- __Modular equation__ solver.
- __Post processing__ export function for viewing results (e.g Paraview).
- __Benchmarking__ tools and diagnostics. 

### Operators
| Name | Description | Syntax | Status |
| --- | --- | --- | --- |
| $\frac{d}{dt}$ | First order time derivative | `solver.ddt_r("U")` | `Release` |
| $\alpha$ . $\nabla^2$ | Laplacian | `solver.laplacian_CSR("U", "Alpha")` | `Release` |
| $\frac{d}{dx}$ | First order spatial derivative | `solver.grad_r("U")` | `Plan` |
***
### Schemes
| Name | Description | Number | Status |
| --- | --- | --- | --- |
| 1st order upwind | Space | 1 | `Plan` |
| 1st order upwind | Space | 2 | `Plan` |
| 1st order central difference | Space | 3 | `Release` |
| Forward Euler | Time | 11 | `Release` |
| Backward Euler | Time | 12 | `Dev` |
| Crank-Nicholson | Time | 13 | `Plan` |
| Leapfrog | Time | 14 | `Plan` |
| RK2 | Time | 15 | `Plan` |
| RK4 | Time | 16 | `Plan` |
| LU Decomposition | Solver | 27 | `Dev` |
***
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Running
1. Download the necessary executables from the latest [release](https://github.com/DjentleViBe/PIRO/releases/new).
2. Copy the executables to the [bin](./bin) directory (create one manually in the root directory if it does not exist).
3. Run the following program to get device details. Refer [Building](#how-to-build) instructions for compiling the program on your own.

| File | Operating Sytem |
| --- | --- |
|```Piro_devices_MacOS``` | MacOS (x64) |
|```Piro_devices_WIN```   | Windows(x64)|
|```Piro_devices_LIN```   | Linux (x64) |

4. Make the required changes to the appropriate files as outlined in [setup.md](./dependencies/assets/setup.md). Make sure all the ```.ini``` files are in the LF format.

5. Execute the command ```./runall.sh -f <setup.ini>``` to initiate the simulation. Log files are stored in the [logs](./logs) folder with the name used for setup file. If the setup file is renamed, update the filename in the command accordingly.

6. Results are stored in the [bin](./bin) directory in a folder with the casename value given in the setup file.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## How to Build
Clone the repo:
```bash
git clone https://github.com/DjentleViBe/PIRO.git
```
## Devices program
1. Switch to the ```devices_list``` branch. 
```bash
git checkout devices_list
``` 
2. Run the command ```./makeall.sh```. The executable file is created in the [bin](./bin) folder.

## Simulation program
1. Switch to the branch of interest using 
```bash
git checkout <branch-name>
``` 
2. Run the command ```./makeall.sh```. The executable file is created in the [bin](./bin) folder.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Documentation
List of manuals:
- [setup.ini](./dependencies/assets/setup.md)
- [IC.ini](./dependencies/assets/IC.md)
- [BC.ini](./dependencies/assets/BC.md)

List of papers:
- [JOSS - A Sparse Matrix parallel solver for OpenCL](./paper.md)

## Feedback and Contributions
Contributions are always welcome to help improve this project! If you have any suggestions, feel free to fork the repository and create a pull request. If you enjoy the project, please consider giving it a star. Thank you for your support!

1. Fork the Project
2. Create your Feature Branch (```git checkout -b feature/AmazingFeature```. For performance studies, ```performance/PerformanceStudyName```.)
3. Commit your Changes (```git commit -m 'Add some AmazingFeature'```)
4. Push to the Branch (```git push origin feature/AmazingFeature```)
5. Open a Pull Request
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License
Distributed under the MIT License.
<p align="right">(<a href="#readme-top">back to top</a>)</p>
