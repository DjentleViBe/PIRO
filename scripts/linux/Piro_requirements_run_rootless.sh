#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
tools_ubuntu_generic=("wget" "git" "clinfo" "rsync" "pciutils" "libpci3" "libkmod2" "pocl" "ocl-icd-opencl-dev" "pocl-opencl-icd" "ocl-icd-libopencl1")
tools_arch=("wget" "git" "clinfo" "rsync" "pciutils" "ocl-icd" "pocl")
#tools_suse_generic=("wget" "git" "which" "rsync" "pciutils" "clinfo" "ocl-icd-devel" "pocl" "pocl-devel" "libOpenCL1")
tools_suse_generic=("pciutils" "clinfo" "ocl-icd-devel" "pocl" "pocl-devel" "libOpenCL1")
tools_fedora_generic=("wget" "git" "which" "clinfo" "rsync" "pciutils" "opencl-headers" "pocl")

architecture=$(uname -m)
bits=$(getconf LONG_BIT)
LOCAL_PREFIX="$(pwd)/local"
mkdir -p "$LOCAL_PREFIX"

if [[ "$architecture" == "x86_64" && "$bits" == "64" ]]; then
    echo "System is 64-bit."
else
    echo "System is NOT x86_64. (detected architecture: $architecture, bits: $bits)"
    exit 1
fi

seen=()

install_ubuntu_package() {
    local pkg="$1"

    if [[ " ${seen[@]} " =~ " $pkg " ]]; then
        return
    fi
    seen+=("$pkg")

    echo "Resolving dependencies for $pkg ..."

    # Create local directories
    mkdir -p "$LOCAL_PREFIX"

    # Get all dependencies (including recommended)
    local all_deps
    all_deps=$(apt-get --just-print install "$pkg" 2>/dev/null \
        | awk '/Inst /{print $2}' | sort -u)

    if [[ -z "$all_deps" ]]; then
        echo "No packages found for $pkg (maybe already downloaded or installed)."
        return
    fi

    echo "Downloading packages: $all_deps"

    # Download all .deb files without installing (non-root)
    apt download $all_deps >/dev/null 2>&1 || {
        echo "Failed to download $pkg or some dependencies"
        return
    }

    # Move downloaded .deb files to LOCAL_PREFIX
    mv -f ./*.deb "$LOCAL_PREFIX"/ 2>/dev/null || true

    echo "Extracting into $LOCAL_PREFIX ..."
    find "$LOCAL_PREFIX" -type f -name "*.deb" | while read -r debfile; do
        dpkg-deb -x "$debfile" "$LOCAL_PREFIX"
    done

    # Optional cleanup
    rm -f "$LOCAL_PREFIX"/*.deb

    # Fix OpenCL ICD files if relevant
    mkdir -p "$LOCAL_PREFIX/etc/OpenCL/vendors"
    find "$LOCAL_PREFIX" -name "*.icd" -exec cp {} "$LOCAL_PREFIX/etc/OpenCL/vendors/" \;

    echo "Done installing $pkg locally."
}

install_arch_package() {
    local root_pkg="$1"
    local -A seen=()
    local -a queue=("$root_pkg")

    echo "Building dependency graph..."

    while ((${#queue[@]})); do
        local pkg="${queue[0]}"
        queue=("${queue[@]:1}")

        [[ -n "${seen[$pkg]}" ]] && continue
        seen["$pkg"]=1

        # Get dependencies for this package
        local deps
        deps=$(pacman -Si "$pkg" 2>/dev/null | awk -F: '/Depends On/ {for (i=3; i<=NF; i++) print $i}' | tr -d ' ' | tr '\n' ' ')

        for d in $deps; do
            [[ -n "$d" && -z "${seen[$d]}" ]] && queue+=("$d")
        done
    done

    local all_pkgs=("${!seen[@]}")
    echo "Total packages to process: ${#all_pkgs[@]}"

    # Download all at once
    pacman -Sw --needed --cachedir "$LOCAL_PREFIX" "${all_pkgs[@]}" --noconfirm

    # Extract all
    for pkg in "${all_pkgs[@]}"; do
        local pkgfile
        pkgfile=$(ls "$LOCAL_PREFIX/$pkg"-*.pkg.tar.zst 2>/dev/null | head -n1)
        if [[ -f "$pkgfile" ]]; then
            echo "Extracting $pkgfile ..."
            bsdtar -xf "$pkgfile" -C "$LOCAL_PREFIX"
        fi
    done
}

install_opensuse_package() {
    local pkg="$1"
    if [[ " ${seen[@]} " =~ " $pkg " ]]; then return; fi
    seen+=("$pkg")

    echo "Resolving dependencies for $pkg ..."

    # Get list of dependencies including pkg itself
    local all_deps
    all_deps=$(zypper --non-interactive install --download-only --dry-run "$pkg" \
        2>/dev/null \
        | awk '/The following [0-9]+ NEW packages/ {flag=1; next} /^$/ {flag=0} flag' \
        | tr -d ',' \
        | awk '{$1=$1; print}')

    if [[ -z "$all_deps" ]]; then
        echo "No new packages to download for $pkg."
        return
    fi

    echo "Downloading packages: $all_deps"
    zypper --non-interactive download $all_deps

    echo "Extracting RPMs into $LOCAL_PREFIX ..."
    find "/var/cache/zypp" -type f -name "*.rpm" | while read -r rpmfile; do
        rpm2cpio "$rpmfile" | (cd "$LOCAL_PREFIX" && cpio -idmv)
    done

    echo "Done installing $pkg locally."
}

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
########################################### UBUNTU ############################################## 
if [ -f /etc/debian_version ]; then
    echo "Debian/Ubuntu detected."
    echo "Downloading and installing missing tools locally..."
    # apt update

    OS_VERSION=$(grep '^VERSION=' /etc/os-release | cut -d'"' -f2)
    check_tools tools_ubuntu_generic missing_tools
    for t in "${missing_tools[@]}"; do
        # echo "Downloading and installing $t ..."
        install_ubuntu_package "$t"
    done
    echo "Setting up environment variables in ~/.bashrc ..."

    grep -qxF "export LOCAL_PREFIX=\"$LOCAL_PREFIX\"" ~/.bashrc || \
        echo "export LOCAL_PREFIX=\"$LOCAL_PREFIX\"" >> ~/.bashrc

    grep -qxF 'export PATH="$LOCAL_PREFIX/usr/bin:$PATH"' ~/.bashrc || \
        echo 'export PATH="$LOCAL_PREFIX/usr/bin:$PATH"' >> ~/.bashrc

    grep -qxF 'export LD_LIBRARY_PATH="$LOCAL_PREFIX/usr/lib:$LOCAL_PREFIX/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"' ~/.bashrc || \
        echo 'export LD_LIBRARY_PATH="$LOCAL_PREFIX/usr/lib:$LOCAL_PREFIX/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"' >> ~/.bashrc

    grep -qxF 'export PKG_CONFIG_PATH="$LOCAL_PREFIX/usr/lib/pkgconfig:$LOCAL_PREFIX/usr/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH"' ~/.bashrc || \
        echo 'export PKG_CONFIG_PATH="$LOCAL_PREFIX/usr/lib/pkgconfig:$LOCAL_PREFIX/usr/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH"' >> ~/.bashrc

    grep -qxF 'export OPENCL_VENDOR_PATH="$LOCAL_PREFIX/etc/OpenCL/vendors"' ~/.bashrc || \
        echo 'export OPENCL_VENDOR_PATH="$LOCAL_PREFIX/etc/OpenCL/vendors"' >> ~/.bashrc

    exec bash
    if [[ "$OS_VERSION" == 22.04* ]]; then
    echo "Ubuntu 22.04 detected."
        if lspci | grep -i nvidia > /dev/null; then
            echo "NVIDIA GPU detected, installing NVIDIA OpenCL ICD..."
            DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
            DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)

            PACKAGE_NAME="libnvidia-compute-${DRIVER_MAJOR}_${DRIVER_VERSION}-0ubuntu0.22.04.4_amd64.deb"
            PACKAGE_URL="https://archive.ubuntu.com/ubuntu/pool/restricted/n/nvidia-graphics-drivers-${DRIVER_MAJOR}/libnvidia-compute-${DRIVER_MAJOR}_${DRIVER_VERSION}-0ubuntu0.22.04.4_amd64.deb"

            # Download the package
            echo "Downloading $PACKAGE_NAME..."
            curl -L -o "$LOCAL_PREFIX/$PACKAGE_URL" || { echo "Download failed"; exit 1; }

            # Install the package
            echo "Installing $PACKAGE_NAME..."
            echo "Extracting $PACKAGE_NAME..."
            dpks -x "$LOCAL_PREFIX/$PACKAGE_NAME" "$LOCAL_PREFIX"
            rm -f "$LOCAL_PREFIX/$PACKAGE_NAME"
            # Delete the package file after install
            echo "Deleting package file $PACKAGE_NAME..."
            rm -f "$PACKAGE_NAME"
            echo "Done."
        else
            echo No NVIDIA GPU detected. Support for additional GPUs for Ubuntu will be added in future releases.
            exit 1
        fi
    else
        if lspci | grep -i nvidia > /dev/null; then
            echo "NVIDIA GPU detected, installing NVIDIA OpenCL ICD..."
            DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
            DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)

            PACKAGE_NAME="libnvidia-compute-${DRIVER_MAJOR}_${DRIVER_VERSION}-0ubuntu0.25.04.4_amd64.deb"
            PACKAGE_URL="https://archive.ubuntu.com/ubuntu/pool/restricted/n/nvidia-graphics-drivers-${DRIVER_MAJOR}/libnvidia-compute-${DRIVER_MAJOR}_${DRIVER_VERSION}-0ubuntu0.25.04.4_amd64.deb"

            # Download the package
            echo "Downloading $PACKAGE_NAME..."
            curl -L -o "$LOCAL_PREFIX/$PACKAGE_URL" || { echo "Download failed"; exit 1; }

            # Install the package
            echo "Installing $PACKAGE_NAME..."
            echo "Extracting $PACKAGE_NAME..."
            dpkg -x "$LOCAL_PREFIX/$PACKAGE_NAME" "$LOCAL_PREFIX"
            rm -f "$LOCAL_PREFIX/$PACKAGE_NAME"
            # Delete the package file after install
            echo "Deleting package file $PACKAGE_NAME..."
            rm -f "$PACKAGE_NAME"
            echo "Done."
        else
            echo No NVIDIA GPU detected. Support for additional GPUs for Ubuntu will be added in future releases.
            exit 1
        fi
    fi

    echo "Environment setup complete."
########################################### FEDORA ##############################################    
elif [ -f /etc/redhat-release ]; then
    release=$(< /etc/redhat-release)
    if [[ "$release" =~ "Fedora" ]]; then
        echo "Fedora detected."
        #for t in "${missing_tools[@]}"; do
        #    recurse_fedora "$t"
        #done
        echo "Fedora support coming sooon..."
        break
    else
        echo "RHEL/CentOS not supported at the moment."
        break
    fi
    grep -qxF "export LOCAL_PREFIX=\"$LOCAL_PREFIX\"" ~/.bashrc || \
    echo "export LOCAL_PREFIX=\"$LOCAL_PREFIX\"" >> ~/.bashrc

    grep -qxF 'export PATH="$LOCAL_PREFIX/usr/bin:$PATH"' ~/.bashrc || \
    echo 'export PATH="$LOCAL_PREFIX/usr/bin:$PATH"' >> ~/.bashrc

    grep -qxF 'export LD_LIBRARY_PATH="$LOCAL_PREFIX/usr/lib64:$LOCAL_PREFIX/lib:$LD_LIBRARY_PATH"' ~/.bashrc || \
    echo 'export LD_LIBRARY_PATH="$LOCAL_PREFIX/usr/lib64:$LOCAL_PREFIX/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc

    grep -qxF 'export PKG_CONFIG_PATH="$LOCAL_PREFIX/usr/lib/pkgconfig:$LOCAL_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"' ~/.bashrc || \
    echo 'export PKG_CONFIG_PATH="$LOCAL_PREFIX/usr/lib/pkgconfig:$LOCAL_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"' >> ~/.bashrc

    grep -qxF 'export OPENCL_VENDOR_PATH="$LOCAL_PREFIX/etc/OpenCL/vendors"' ~/.bashrc || \
    echo 'export OPENCL_VENDOR_PATH="$LOCAL_PREFIX/etc/OpenCL/vendors"' >> ~/.bashrc
    exec bash
    #$SUDO ln -s $(which g++-14) /usr/bin/g++
    #$SUDO ln -sf /lib64/libOpenCL.so.1 /lib64/libOpenCL.so
    if lspci | grep -i nvidia > /dev/null; then
        if rpm -q xorg-x11-drv-nvidia &>/dev/null; then
            echo "NVIDIA proprietary driver package is installed."
        else
            echo "NVIDIA GPU detected, installing NVIDIA OpenCL..."
            wget https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
            wget https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
            mkdir -p ~/nvidia-offline
            cd ~/nvidia-offline
            dnf download --resolve xorg-x11-drv-nvidia-cuda
        fi
    elif lspci | grep -i amd > /dev/null; then
        echo "AMD GPU detected, installing AMD OpenCL..."
        yum install -y rocm-opencl
    fi
########################################### ARCH ############################################## 
elif [ -f /etc/arch-release ]; then
    echo "Arch Linux detected."
    check_tools tools_arch missing_tools

    echo "Downloading and installing missing tools locally..."

    for t in "${missing_tools[@]}"; do
        # echo "Downloading and installing $t ..."
            install_arch_package "$t"
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
        curl -L -o "$LOCAL_PREFIX/$PACKAGE_NAME" "$PACKAGE_URL" || { echo "Download failed"; exit 1; }

        # Install the package
        echo "Installing $PACKAGE_NAME..."
        echo "Extracting $PACKAGE_NAME..."
        tar -I zstd -xf "$LOCAL_PREFIX/$PACKAGE_NAME" -C "$LOCAL_PREFIX"
        rm -f "$LOCAL_PREFIX/$PACKAGE_NAME"
        # Delete the package file after install
        echo "Deleting package file $PACKAGE_NAME..."
        rm -f "$PACKAGE_NAME"
        echo "Done."
    elif lspci | grep -i amd > /dev/null; then
        echo "AMD GPU detected, installing AMD OpenCL..."
        PACKAGE_NAME="rocm-opencl-runtime-6.4.4-1-x86_64.pkg.tar.zst"
        PACKAGE_URL="https://archive.archlinux.org/packages/r/rocm-opencl-runtime/${PACKAGE_NAME}"
        echo "Downloading $PACKAGE_NAME..."
        curl -L -o "$LOCAL_PREFIX/$PACKAGE_NAME" "$PACKAGE_URL" || { echo "Download failed"; exit 1; }
        echo "Installing $PACKAGE_NAME..."
        echo "Extracting $PACKAGE_NAME..."
        tar -I zstd -xf "$LOCAL_PREFIX/$PACKAGE_NAME" -C "$LOCAL_PREFIX"
        rm -f "$LOCAL_PREFIX/$PACKAGE_NAME"
        echo "Deleting package file $PACKAGE_NAME..."
        rm -f "$PACKAGE_NAME"
    elif lspci | grep -i intel > /dev/null; then
        echo "INTEL GPU detected, installing INTEL OpenCL..."
        PACKAGE_NAME="intel-compute-runtime-25.35.35096.9-1-x86_64.pkg.tar.zst"
        PACKAGE_URL="https://archive.archlinux.org/packages/i/intel-compute-runtime/${PACKAGE}"
        echo "Downloading $PACKAGE_NAME..."
        curl -L -o "$LOCAL_PREFIX/$PACKAGE_NAME" "$PACKAGE_URL" || { echo "Download failed"; exit 1; }
        echo "Installing $PACKAGE_NAME..."
        echo "Extracting $PACKAGE_NAME..."
        tar -I zstd -xf "$LOCAL_PREFIX/$PACKAGE_NAME" -C "$LOCAL_PREFIX"
        rm -f "$LOCAL_PREFIX/$PACKAGE_NAME"
        echo "Deleting package file $PACKAGE_NAME..."
        rm -f "$PACKAGE_NAME"
        echo "Intel OpenCL support can be a bit tricky — intel-compute-runtime is usually good but sometimes the newer 
        opencl-intel + related packages are needed for newer hardware. Confirm hardware detection 
        by running clinfo"
    fi
    echo "Setting up environment variables in ~/.bashrc ..."
    grep -qxF "export LOCAL_PREFIX=\"$LOCAL_PREFIX\"" ~/.bashrc || \
    echo "export LOCAL_PREFIX=\"$LOCAL_PREFIX\"" >> ~/.bashrc
    grep -qxF "export PATH=\"$LOCAL_PREFIX/usr/bin:\$PATH\"" ~/.bashrc || \
    echo "export PATH=\"$LOCAL_PREFIX/usr/bin:\$PATH\"" >> ~/.bashrc
    grep -qxF "export LD_LIBRARY_PATH=\"$LOCAL_PREFIX/usr/lib:\$LD_LIBRARY_PATH\"" ~/.bashrc || \
    echo "export LD_LIBRARY_PATH=\"$LOCAL_PREFIX/usr/lib:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
    grep -qxF "export PKG_CONFIG_PATH=\"$LOCAL_PREFIX/usr/lib/pkgconfig:\$PKG_CONFIG_PATH\"" ~/.bashrc || \
    echo "export PKG_CONFIG_PATH=\"$LOCAL_PREFIX/usr/lib/pkgconfig:\$PKG_CONFIG_PATH\"" >> ~/.bashrc
    grep -qxF "export OPENCL_VENDOR_PATH="$LOCAL_PREFIX/etc/OpenCL/vendors"" ~/.bashrc || \
    echo "export OPENCL_VENDOR_PATH="$LOCAL_PREFIX/etc/OpenCL/vendors"" >> ~/.bashrc

    exec bash
    echo "Environment setup complete."
########################################### SUSE ############################################## 
elif [ -f /etc/SuSE-release ] || grep -qi "opensuse" /etc/os-release; then
    echo "openSUSE detected."
    check_tools tools_suse_generic missing_tools
    for t in "${missing_tools[@]}"; do
        # echo "Downloading and installing $t ..."
        install_opensuse_package "$t"
    done

    echo "Setting up environment variables in ~/.bashrc ..."
    grep -qxF "export LOCAL_PREFIX=\"$LOCAL_PREFIX\"" ~/.bashrc || \
    echo "export LOCAL_PREFIX=\"$LOCAL_PREFIX\"" >> ~/.bashrc

    grep -qxF 'export PATH="$LOCAL_PREFIX/usr/bin:$PATH"' ~/.bashrc || \
    echo 'export PATH="$LOCAL_PREFIX/usr/bin:$PATH"' >> ~/.bashrc

    grep -qxF 'export LD_LIBRARY_PATH="$LOCAL_PREFIX/usr/lib64:$LOCAL_PREFIX/lib:$LD_LIBRARY_PATH"' ~/.bashrc || \
    echo 'export LD_LIBRARY_PATH="$LOCAL_PREFIX/usr/lib64:$LOCAL_PREFIX/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc

    grep -qxF 'export PKG_CONFIG_PATH="$LOCAL_PREFIX/usr/lib/pkgconfig:$LOCAL_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"' ~/.bashrc || \
    echo 'export PKG_CONFIG_PATH="$LOCAL_PREFIX/usr/lib/pkgconfig:$LOCAL_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"' >> ~/.bashrc

    grep -qxF 'export OPENCL_VENDOR_PATH="$LOCAL_PREFIX/etc/OpenCL/vendors"' ~/.bashrc || \
    echo 'export OPENCL_VENDOR_PATH="$LOCAL_PREFIX/etc/OpenCL/vendors"' >> ~/.bashrc
    exec bash
    echo "Environment setup complete."
    if lspci | grep -i nvidia > /dev/null; then
    # Detect distro
        if grep -q "openSUSE Tumbleweed" /etc/os-release; then
            REPO_URL="https://download.nvidia.com/opensuse/tumbleweed"
        else
            VERSION_ID=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)
            REPO_URL="https://download.nvidia.com/opensuse/leap/$VERSION_ID"
        fi

        # Add the repo if not already added
        if ! zypper repos | grep -q "NVIDIA"; then
            zypper addrepo --no-gpgcheck --refresh "$REPO_URL" NVIDIA
        fi

        # Refresh repo metadata
        zypper refresh

        if zypper search --installed-only x11-video-nvidiaG05 | grep -q x11-video-nvidiaG05; then
            echo "NVIDIA proprietary driver package is installed."
        else
            echo "NVIDIA GPU detected, installing NVIDIA OpenCL..."
            zypper download --repo NVIDIA nvidia-computeG05 --destdir "$LOCAL_PREFIX"
            for rpm in "$LOCAL_PREFIX/"*.rpm; do
                rpm2cpio "$rpm" | (cd "$LOCAL_PREFIX" && cpio -idmv)
            done
        fi
    else 
        echo No NVIDIA GPU detected. Support for additional GPUs for OpenSUSE will be added in future releases.
        exit 1
    fi
    
    echo "Environment setup complete."
else
    echo "Unsupported Linux distribution. Please run this script on Debian/Ubuntu, Fedora, Arch Linux, or openSUSE."
    exit 1
fi

# Run the program
git config --global core.autocrlf false
chmod +x $SCRIPT_DIR/PIRO_devices_LIN
