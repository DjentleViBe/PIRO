#!/bin/bash

cp -r ./dependencies/assets ./bin/.
mkdir -p logs

detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ./bin/PIRO_devices_LIN &> ./logs/devices.txt
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        ./bin/PIRO_devices_MACOS &> ./logs/devices.txt
    elif [[ "$OSTYPE" == "cygwin" ]]; then
        ./bin/PIRO_devices_WIN &> ./logs/devices.txt
    elif [[ "$OSTYPE" == "msys" ]]; then
        ./bin/PIRO_devices_WIN &> ./logs/devices.txt
    elif [[ "$OSTYPE" == "win32" ]]; then
        ./bin/PIRO_devices_WIN &> ./logs/devices.txt
    elif [[ "$OSTYPE" == "freebsd"* ]]; then
        echo "Operating System: FreeBSD"
    else
        echo "Operating System: Unknown"
    fi
}
detect_os

