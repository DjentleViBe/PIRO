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
