# setup.ini
This section is used to give details on fields and parameters used in setup.ini files. Please refer to the Features section on the main page for latest implementation before changing values.

## [File]
Folder and file management section.

### casename
The casename is used for naming the folder and consequent files which are exported as results. 

### restart
If restart = 1, the folder which is created with the ```filename``` is deleted and recreated.

## [Device]
This section sets the device used for the simulation. The following parameters can be fetched by running the following file:

Table: Executable files for different Operating Systems.

+-------------------------+--------------------------+
| **Filename**            |**Operating System**      |
+=========================+==========================+
|```Piro_devices_MacOS``` | MacOS (x64)              |
+-------------------------+--------------------------+
|```Piro_devices_WIN```   | Windows(x64)             |
+-------------------------+--------------------------+
|```Piro_devices_LIN```   | Linux (x64)              |
+-------------------------+--------------------------+

### id
Set the id parameter retreived after running the file from Table 1.

### platformid
Set the platformid retreived after running the file from Table 1.

## type
Mode 0 runs on CPU (serial), mode 1 on CPU (parallel), and mode 2 on GPU (parallel).

## [Mesh]
Mesh settings

### Mesh Type
0 for Cartesian grid

### o
Sets the origin of the block in ```x, y, z``` coordinates.

### n
Sets the number of cartesian grids in ```x, y, z``` direction.

### l
Sets the size of the block in ```x, y, z``` direction.

### s
Sets the scale of the block in ```x, y, z``` direction.

## [Simulation]
Parameters used for running the simulation. Multiple values / variables / fields should be separated by whitespace.

### Constants
Constants to be used the equations go here. For example,
```Constants = Alpha Hbar2m```

### Values
Set the values for the constants from the previous section. For example,
```Values = 0.1```

### Scalars
Scalar fileds to be used in the equations go here. For example,
```Scalars = P```

### Vectors
Vector fields to be used in the equations go here. For example,
```Vectors = U```

## [IC]
The section initialises the constants and variables used in the equations.

### type
Mode 0 sets all the values from the values section, Mode 1 sets values from a ```*.ini``` file and Mode 2 sets values from a ```.csv``` file. Multiple entries should be seperted by whitespace and needs to be ordered as 

Scalar_1, Scalar_2...Scalar_n, Vector_1, Vector_2...Vector_n.

### filename
For Modes 1 and 2, this takes filenames for the Scalars and Vectors. File details for mode 1: ```filename```.ini under ```dependencies/assets/IC```.

File details for mode 2: ```filename```.csv under ```dependencies/assets/IC```

## [BC]
This sections gives details on setting Boundary conditions for the domain block.
Formatting follows as explained under [IC] section

### type
Mode 0 : Dirichlet, Mode 1 : Neumann, Mode 2: Robin, Mode 3: Mixed, Mode 4: Cauchy
Faces order : YZ, YZ + l, XZ + b, XZ, XY, XY + h.

### property
Use Variable names from [Simulation section].

### values
set a constant value, use a .csv file from or .ini file from ```dependencies/assets/BC```. If the property is a vector, the values should be serialised. 

## [Schemes]
Table: Spacial schemes

+-------------------------+-------------------------------+
| **Mode**                |**Scheme**                     |
+=========================+===============================+
|```1```                  | 1st order upwind              |
+-------------------------+-------------------------------+
|```2```                  | 1st order downwind            |
+-------------------------+-------------------------------+
|```3```                  | 1st order central difference  |
+-------------------------+-------------------------------+
|```4```                  | Lax-Wendroff                  |
+-------------------------+-------------------------------+

Table: Temporal schemes

+-------------------------+-------------------------------+
| **Mode**                |**Scheme**                     |
+=========================+===============================+
|```11```                 | Forward Euler                 |
+-------------------------+-------------------------------+
|```12```                 | Backward Euler                |
+-------------------------+-------------------------------+
|```13```                 | Crank-Nicholson               |
+-------------------------+-------------------------------+
|```14```                 | Leapfrog                      |
+-------------------------+-------------------------------+
|```15```                 | RK2                           |
+-------------------------+-------------------------------+
|```16```                 | RK4                           |
+-------------------------+-------------------------------+


Table: Matrix Solver

+-------------------------+-------------------------------+
| **Mode**                |**Scheme**                     |
+=========================+===============================+
|```17```                 | Implicit LU decomposition     |
+-------------------------+-------------------------------+

## [Solve]
Parameters used for solving equations.

### Data
Data formats to be sued for storing matrices:
0 - Compressed Sparse Row (CSR).
1 - Dense matrix.
2 - Hash tables.

### Timestep
Delta time used for time stepping in Implicit schemes.

### Total Time
Total time for running simultions in Implicit schemes.

### Iterations
Total number of iterations for running simulations in Expliit schemes

### Save
Iteration / Timestep frequency to write results to disk.

## [PostProcess]
Settings for exported results.

### RunICS
Mode 0 exports to Paraview in ```.vth``` format. Timesteps are exported in ```.vti``` format.

## [Debug]
Settings for logger.

### Verbose
Mode 0 - Information 

Mode 1 - Debug

Mode 2 - Warning

Mode 3 - Errors

Mode 4 - Trace

