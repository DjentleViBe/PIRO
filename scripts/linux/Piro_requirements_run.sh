#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
tools_ubuntu=("git" "clinfo" "rsync" "pciutils" "ocl-icd-opencl-dev" "pocl-opencl-icd")
tools_suse=("git" "clinfo" "rsync" "pciutils" "ocl-icd-devel" "pocl" "pocl-devel")
tools_fedora=("git" "which" "clinfo" "rsync" "pciutils" "opencl-headers" "pocl")
tools_arch=("git" "clinfo" "rsync" "pciutils" "ocl-icd-opencl-dev")

# Run as root check
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo $0"
    exit 1
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
        # apt install -y libcompute-nvidia-570
    fi

elif [ -f /etc/redhat-release ]; then
    echo "Red Hat/CentOS/Fedora detected."
    yum update -y
    check_tools tools_fedora missing_tools
    for t in "${missing_tools[@]}"; do
        yum install -y "$t"
    done

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

    if lspci | grep -i nvidia > /dev/null; then
        echo "NVIDIA GPU detected, installing NVIDIA OpenCL..."
        # zypper install -y opencl-nvidia
    fi
fi

# Run the program
git config --global core.autocrlf false
chmod +x $SCRIPT_DIR/PIRO_devices_LIN
