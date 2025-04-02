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
- [About](#-about)
- [Features](#-certification)
- [How to Build](#-how-to-build)
- [Documentation](#-documentation)
- [Feedback and Contributions](#-feedback-and-contributions)
- [License](#-license)
- [Contacts](#%EF%B8%8F-contacts)

## About 
## Features
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
***
`Plan Dev Release`
## Running
### Windows
1. Run ```.\bin\GIRO_devices.exe``` to get list of platform id and device id.
2. Edit id and platformid values in .\dependencies\assets\setup_*.ini. with the numbers obtained from 1.
3. Make sure the following files are in the LF format.
    - ./dependencies/assets/IC/*.ini
    - ./dependencies/assets/*.ini
4. Run ```perf_test.bat```

### Macos
1. Run ```./bin/GIRO_devices``` to get list of platform id and device id.
2. Edit id and platformid values in ./dependencies/assets/setup_*.ini. with the numbers obtained from 1.
3. Make sure the following files are in the LF format.
    - ./dependencies/assets/IC/*.ini
    - ./dependencies/assets/*.ini
4. Run ```./perf_test.sh```

Logs are output inside ./logs/
## How to Build
### Requirements
```shell
```
## Documentation
## Feedback and Contributions
