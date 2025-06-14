#!/bin/bash

# Run as root check
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo $0"
    exit 1
fi

if [ -f /etc/debian_version ]; then
    echo "Debian/Ubuntu detected."
    apt update
    apt install -y git clinfo ocl-icd-opencl-dev

    # Optional: NVIDIA OpenCL runtime if NVIDIA GPU detected
    if lspci | grep -i nvidia > /dev/null; then
        echo "NVIDIA GPU detected, installing NVIDIA OpenCL ICD..."
        apt install -y nvidia-opencl-icd
    fi

elif [ -f /etc/redhat-release ]; then
    echo "Red Hat/CentOS/Fedora detected."
    yum install -y git clinfo ocl-icd-devel

    if lspci | grep -i nvidia > /dev/null; then
        echo "NVIDIA GPU detected, installing NVIDIA OpenCL..."
        yum install -y opencl-nvidia
    fi

elif [ -f /etc/arch-release ]; then
    echo "Arch Linux detected."
    pacman -Sy --noconfirm git clinfo ocl-icd

    if lspci | grep -i nvidia > /dev/null; then
        echo "NVIDIA GPU detected, installing NVIDIA OpenCL..."
        pacman -S --noconfirm opencl-nvidia
    elif lspci | grep -i amd > /dev/null; then
        echo "AMD GPU detected, installing AMD OpenCL..."
        pacman -S --noconfirm opencl-amd
    elif lspci | grep -i intel > /dev/null; then
        echo "INTEL GPU detected, installing INTEL OpenCL..."
        pacman -S --noconfirm opencl-intel
        echo "Intel OpenCL support can be a bit tricky — opencl-intel is usually good but sometimes the newer 
        intel-compute-runtime + related packages are needed for newer hardware. Confirm hardware detection 
        by running clinfo"
    fi

elif [ -f /etc/SuSE-release ] || grep -qi "opensuse" /etc/os-release; then
    echo "openSUSE detected."
    zypper refresh
    zypper install -y git clinfo ocl-icd-devel

    if lspci | grep -i nvidia > /dev/null; then
        echo "NVIDIA GPU detected, installing NVIDIA OpenCL..."
        zypper install -y opencl-nvidia
    fi
fi

# Run the program
git config --global core.autocrlf false
chmod +x ./PIRO_devices_LIN
echo "Please note down the Device and Platform number of the hardware where the simulation is intended to be run."
./PIRO_devices_LIN