#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
tools_ubuntu_2204=("wget" "git=1:2.34.1-1ubuntu1.15" "make=4.3-4.1build1" "gcc=4:11.2.0-1ubuntu1" "g++=4:11.2.0-1ubuntu1" "clinfo=3.0.21.02.21-1" "rsync" "pciutils" "ocl-icd-opencl-dev=2.2.14-3" "pocl-opencl-icd=1.8-3")
tools_ubuntu=("wget" "git" "make" "gcc" "g++" "clinfo" "rsync" "pciutils" "ocl-icd-opencl-dev" "pocl-opencl-icd")
tools_suse=("wget" "git-2.50.1-2.1" "make-4.4.1-3.3" "which" "gcc14" "gcc14-c++" "clinfo-3.0.25.02.14-1.1" "rsync" "pciutils" "ocl-icd-devel-2.3.3-1.1" "pocl-7.0-1.1" "pocl-devel-7.0-1.1")
tools_suse_generic=("wget" "git" "make" "which" "gcc14" "gcc14-c++" "clinfo" "rsync" "pciutils" "ocl-icd-devel" "pocl" "pocl-devel")
tools_fedora_42=("wget" "git-2.50.1-1.fc42" "which" "make-1:4.4.1-10.fc42" "gcc14" "gcc14-c++" "clinfo-3.0.23.01.25-7.fc42" "rsync" "pciutils" "opencl-headers-3.0-32.20241023git4ea6df1.fc42" "pocl-6.0-6.fc42")
tools_fedora_generic=("wget" "git" "which" "make" "gcc14" "gcc14-c++" "clinfo" "rsync" "pciutils" "opencl-headers" "pocl")
tools_arch=("wget" "git" "make" "gcc" "g++" "clinfo" "rsync" "pciutils" "ocl-icd")

if command -v sudo &>/dev/null; then
    SUDO="sudo"
else
    SUDO=""
fi

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
    OS_VERSION=$(grep '^VERSION=' /etc/os-release | cut -d'"' -f2)
    if [[ "$OS_VERSION" == 22.04* ]]; then
        check_tools tools_ubuntu_2204 missing_tools
    else
        check_tools tools_ubuntu_generic missing_tools
    fi
    for t in "${missing_tools[@]}"; do
        apt install -y "$t"
    done

    if lspci | grep -i nvidia > /dev/null; then
        echo "NVIDIA GPU detected, installing NVIDIA OpenCL ICD..."
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
        DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
        apt install -y "libnvidia-compute-${DRIVER_MAJOR}"
    else
        echo No NVIDIA GPU detected. Support for additional GPUs for Ubuntu will be added in future releases.
        exit 1
    fi
    

elif [ -f /etc/redhat-release ]; then
    echo "Red Hat/CentOS/Fedora detected."
    yum update -y
    OS_VERSION=$(grep '^VERSION_ID=' /etc/os-release | cut -d'=' -f2 | tr -d '"')
    if [[ "$OS_VERSION" == "42" ]]; then
        check_tools tools_fedora_42 missing_tools
    else
        check_tools tools_fedora_generic missing_tools
    fi
    for t in "${missing_tools[@]}"; do
        yum install -y "$t"
    done
    $SUDO ln -s $(which g++-14) /usr/bin/g++
    $SUDO ln -sf /lib64/libOpenCL.so.1 /lib64/libOpenCL.so
    if lspci | grep -i nvidia > /dev/null; then
        if rpm -q xorg-x11-drv-nvidia &>/dev/null; then
            echo "NVIDIA proprietary driver package is installed."
        else
            echo "NVIDIA GPU detected, installing NVIDIA OpenCL..."
            $SUDO dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
            $SUDO dnf install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
            dnf install -y --allowerasing xorg-x11-drv-nvidia-cuda
        fi
    elif lspci | grep -i amd > /dev/null; then
        echo "AMD GPU detected, installing AMD OpenCL..."
        yum install -y rocm-opencl
    fi

elif [ -f /etc/arch-release ]; then
    echo "Arch Linux detected."
    check_tools tools_arch missing_tools
    for t in "${missing_tools[@]}"; do
        pacman -Sy --noconfirm "$t"
    done

    if lspci | grep -i nvidia > /dev/null; then
        echo "NVIDIA GPU detected, installing NVIDIA OpenCL..."
        # Extract driver version, e.g. "570.153.02"
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)

        if [ -z "$DRIVER_VERSION" ]; then
            echo "Could not detect NVIDIA driver version."
            exit 1
        fi

        echo "Detected NVIDIA driver version: $DRIVER_VERSION"

        # Compose package filename and URL
        PACKAGE_NAME="opencl-nvidia-${DRIVER_VERSION}-1-x86_64.pkg.tar.zst"
        PACKAGE_URL="https://archive.archlinux.org/packages/o/opencl-nvidia/${PACKAGE_NAME}"

        # Download the package
        echo "Downloading $PACKAGE_NAME..."
        wget -q "$PACKAGE_URL" || { echo "Download failed"; exit 1; }

        # Install the package
        echo "Installing $PACKAGE_NAME..."
        $SUDO pacman -U --noconfirm "$PACKAGE_NAME" --overwrite usr/lib/libnvidia-opencl.so\* || { echo "Installation failed"; rm -f "$PACKAGE_NAME"; exit 1; }

        # Delete the package file after install
        echo "Deleting package file $PACKAGE_NAME..."
        rm -f "$PACKAGE_NAME"

        echo "Done."
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
    if grep -q "openSUSE Tumbleweed" /etc/os-release; then
        check_tools tools_suse missing_tools
    else
        check_tools tools_suse_generic missing_tools
    fi
    for t in "${missing_tools[@]}"; do
        zypper install -y "$t"
    done
    $SUDO ln -s $(which g++-14) /usr/bin/g++
    if lspci | grep -i nvidia > /dev/null; then
        if zypper search --installed-only x11-video-nvidiaG05 | grep -q x11-video-nvidiaG05; then
            echo "NVIDIA proprietary driver package is installed."
        else
            echo "NVIDIA GPU detected, installing NVIDIA OpenCL..."
            if grep -q "openSUSE Tumbleweed" /etc/os-release; then
                zypper addrepo --no-gpgcheck --refresh https://download.nvidia.com/opensuse/tumbleweed NVIDIA
            else
                VERSION_ID=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)
                zypper addrepo --no-gpgcheck --refresh https://download.nvidia.com/opensuse/leap/$VERSION_ID NVIDIA
            fi
            zypper refresh
            zypper install nvidia-computeG05
        fi
    else 
        echo No NVIDIA GPU detected. Support for additional GPUs for OpenSUSE will be added in future releases.
        exit 1
    fi
else
    echo "Unsupported Linux distribution. Please run this script on Debian/Ubuntu, Fedora, Arch Linux, or openSUSE."
    exit 1
fi

# Run the program
git config --global core.autocrlf false
chmod +x ./makedevice.sh
