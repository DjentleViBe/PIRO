#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
tools_ubuntu=("git" "make" "gcc" "g++" "clinfo" "rsync" "pciutils" "git" "ocl-icd-opencl-dev")
tools_suse=("git" "make" "gcc14" "gcc14-c++" "clinfo" "rsync" "pciutils" "git" "ocl-icd-devel")
tools_fedora=("git" "which" "make" "gcc14" "gcc14-c++" "clinfo" "rsync" "pciutils" "git" "opencl-headers")
tools_arch=("git" "make" "gcc" "g++" "clinfo" "rsync" "pciutils" "git" "ocl-icd-opencl-dev")

if command -v sudo &>/dev/null; then
    SUDO="sudo"
else
    SUDO=""
fi

missing_tools=()
echo "Checking commands..."

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

check_tools() {
    local -n tools_ref=$1          # input array of tools to check
    local -n missing_ref=$2        # output array name to store missing tools
    missing_ref=()                 # clear output array

    for tool in "${tools_ref[@]}"; do
        if command_exists "$tool"; then
            echo "$tool is installed."
        else
            echo "$tool is NOT installed."
            missing_ref+=("$tool")
        fi
    done

    if [ ${#missing_ref[@]} -gt 0 ]; then
        echo "Missing tools: ${missing_ref[*]}"
    else
        echo "All tools are installed."
    fi
}

# Run as root check
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo $0"
    exit 1
fi

if [ -f /etc/debian_version ]; then
    echo "Debian/Ubuntu detected."
    apt update
    check_tools tools_ubuntu missing_tools
    for t in "${missing_tools[@]}"; do
        apt install -y "$t"
    done

    # Optional: NVIDIA OpenCL runtime if NVIDIA GPU detected
    if lspci | grep -i nvidia > /dev/null; then
        echo "NVIDIA GPU detected, installing NVIDIA OpenCL ICD..."
        # apt install -y libnvidia-compute-570
    fi

elif [ -f /etc/redhat-release ]; then
    echo "Fedora detected."
    yum update -y
    check_tools tools_fedora missing_tools
    for t in "${missing_tools[@]}"; do
        yum install --assumeyes "$t"
    done
    $SUDO ln -s $(which g++-14) /usr/bin/g++
    $SUDO ln -sf /lib64/libOpenCL.so.1 /lib64/libOpenCL.so

    if lspci | grep -i nvidia > /dev/null; then
        echo "NVIDIA GPU detected, installing NVIDIA OpenCL..."
        # yum install -y opencl-nvidia
    fi

elif [ -f /etc/arch-release ]; then
    echo "Arch Linux detected."
     check_tools tools_arch missing_tools
    for t in "${missing_tools[@]}"; do
        pacman -Sy --noconfirm "$t"
    done

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
    check_tools tools_suse missing_tools
    for t in "${missing_tools[@]}"; do
        zypper install -y "$t"
    done
    $SUDO ln -s $(which g++-14) /usr/bin/g++
    if lspci | grep -i nvidia > /dev/null; then
        echo "NVIDIA GPU detected, installing NVIDIA OpenCL..."
        # zypper install -y opencl-nvidia
    fi
fi

# Run the program
git config --global core.autocrlf false
chmod +x ./makedevice.sh
