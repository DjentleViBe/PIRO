#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
tools_ubuntu=("git=1:2.34.1-1ubuntu1.15" "make=4.3-4.1build1" "gcc=4:11.2.0-1ubuntu1" "g++=4:11.2.0-1ubuntu1" "clinfo=clinfo=3.0.21.02.21-1" "rsync" "pciutils" "ocl-icd-opencl-dev=2.2.14-3" "pocl-opencl-icd=1.8-3")
tools_suse=("git-2.50.1-2.1" "make-4.4.1-3.3" "which" "gcc14" "gcc14-c++" "clinfo-3.0.25.02.14-1.1" "rsync" "pciutils" "ocl-icd-devel-2.3.3-1.1" "pocl-7.0-1.1" "pocl-devel-7.0-1.1")
tools_fedora=("git-2.50.1-1.fc42" "which" "make-1:4.4.1-10.fc42" "gcc14" "gcc14-c++" "clinfo-3.0.23.01.25-7.fc42" "rsync" "pciutils" "opencl-headers-3.0-32.20241023git4ea6df1.fc42" "pocl-6.0-6.fc42")
tools_arch=("git" "make" "gcc" "g++" "clinfo" "rsync" "pciutils" "ocl-icd")

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
    local -n tools_ref=$1
    local -n missing_ref=$2
    missing_ref=()

    for tool in "${tools_ref[@]}"; do
        # Strip everything after the first '=' to get the command name
        if [[ "$tool" == *"="* ]]; then
            cmd="${tool%%=*}"       # Debian/Ubuntu
        elif [[ "$tool" =~ -[0-9] ]]; then
            cmd="${tool%%-[0-9]*}"  # Fedora/RHEL style
        else
            cmd="$tool"             # No version specified
        fi
        if command_exists "$cmd"; then
            echo "$cmd is installed."
        else
            echo "$cmd is NOT installed."
            missing_ref+=("$tool") # Keep original string with version
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
