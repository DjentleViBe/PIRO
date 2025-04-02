# **G**pu dr**I**ven nume**R**ical s**O**lver

## Running Performance test

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

Logs are output inside logs folder


### FLOPS calculations:
## LU_Decomp_part2:
Number of work items = 8
Floating Point Operations :
1. kernelfilterarray = 0
2. kernelfilterrow = 2
3. kernel_math[1] = 1

Iteration 0 = kernelfilterarray = 0
Iteration 1- 7 = 7*(kernelfilterarray + kernelfilterrow + kernel_math[1]) = 7 * 3 = 21

Total FLOPS = 8 * 21 = 168
Execution time = 0.0007 * 8 * 8 = 0.0448 (0.0790 total)
GFLOPS = 168 / 0.0448 = 3.75E-6

Ops per clock = Peak FLOPS / Clock speed = 6.4E12 / 1E9 = 6.4E3
Peak GFLOPS = Compute Units * Clock Speed * Ops per clock / 1E9 = 20 * 6.4E3 * 1E9 / 1E9 = 128E3

Efficiency = GFLOPS / PeakGFLOPS = 2545.45 / 128E3 = 0.0292

## LU_Decomp_part1:
1. lu_decompose_dense = 3

Iteration 0-7 = 8*(lu_decompose_dense) = 8 * 3 = 24
Total FLOPS = 8 * 24 = 192
Execution time = 0.005
GFLOPS = 192 / 0.005 = 38400

Ops per clock = 6.4E3
Peak GFLOPS = 128E3

Efficiency = 38400 / 128E3 = 0.3

## LU_Decomp_part4:
1. kernelfilterarray = 3 in best case, 4 in worst case

Iteration 0-7 = 8*(kernelfilterarray) = 8 * 4 = 32
Total FLOPS = 8 * 32 = 256
Execution time = 0.00001 + 7 * (0.000051 + 0.000045 + 0.001024) = 0.00112 (0.03457 total)
GFLOPS = 256 / 0.00112 = 228571

Ops per clock = 228571
Peak GFLOPS = 128E3

Efficiency = 11428 / 128E3 = 1.943
