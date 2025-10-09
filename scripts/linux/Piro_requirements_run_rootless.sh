#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
tools_ubuntu_generic=("wget" "git" "rsync" "clinfo" "make" "build-essential" "g++" "libc6" "libc6-dev" "pciutils" "ocl-icd-opencl-dev")
tools_arch=("wget" "git" "clinfo" "rsync" "pciutils" "ocl-icd" "pocl")
tools_suse_generic=("wget" "git" "which" "rsync" "pciutils" "clinfo" "ocl-icd-devel" "pocl" "pocl-devel" "libOpenCL1")
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
        echo "Package $pkg already installed on host; forcing local extraction."
        all_deps="$pkg"
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
    # find "$LOCAL_PREFIX" -name "*.icd" -exec cp {} "$LOCAL_PREFIX/etc/OpenCL/vendors/" \;

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
    
    echo "Resolving complete dependency tree for $pkg ..."

    # Get COMPLETE list (zypper does the recursion for you)
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

    echo "Downloading all packages: $all_deps"
    zypper --non-interactive download $all_deps

    echo "Extracting RPMs into $LOCAL_PREFIX ..."
    find "/var/cache/zypp" -type f -name "*.rpm" | while read -r rpmfile; do
        rpm2cpio "$rpmfile" | (cd "$LOCAL_PREFIX" && cpio -idmv)
    done

    # Patch ICD files in both possible locations:
    for icd_dir in "$LOCAL_PREFIX/etc/OpenCL/vendors" "$LOCAL_PREFIX/usr/share/OpenCL/vendors"; do
        if [[ -d "$icd_dir" ]]; then
            echo "Patching OpenCL ICD files in $icd_dir..."
            find "$icd_dir" -name "*.icd" | while read -r icd; do
                sed -i "s|^/usr|$LOCAL_PREFIX/usr|g" "$icd"
            done
        fi
    done

    echo "Done!"
}

missing_tools=()
echo "Checking commands..."

command_exists_in_sysroot() {
    local cmd="$1"
    # Check in sysroot's bin directories
    if [ -f "$LOCAL_PREFIX/usr/bin/$cmd" ] || [ -f "$LOCAL_PREFIX/bin/$cmd" ]; then
        return 0
    fi
    return 1
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
        if command_exists_in_sysroot "$cmd"; then
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
    mkdir -p "$LOCAL_PREFIX/lib/x86_64-linux-gnu"
    cp -r $LOCAL_PREFIX/usr/lib/ $LOCAL_PREFIX/
    mkdir -p "$LOCAL_PREFIX/lib64/"
    cp -r $LOCAL_PREFIX/usr/lib64/ $LOCAL_PREFIX/
    mkdir -p $LOCAL_PREFIX/usr/lib/x86_64-linux-gnu
    cp /usr/lib/x86_64-linux-gnu/libgcc_s.so.1 $LOCAL_PREFIX/usr/lib/x86_64-linux-gnu/
    #mkdir -p $LOCAL_PREFIX/share/pocl/include
    #cp $LOCAL_PREFIX/usr/share/pocl/include/* $LOCAL_PREFIX/share/pocl/include/.
    #cp $LOCAL_PREFIX/usr/share/pocl/* $LOCAL_PREFIX/share/pocl/.

    echo "Setting up environment variables in ~/.bashrc ..."
    BASHRC="$HOME/.bashrc"

    # Marker to avoid duplicates
    MARKER="# --- Local build environment setup ---"
    if ! grep -Fxq "$MARKER" "$BASHRC"; then
        cat << EOF >> "$BASHRC"

# --- Local build environment setup ---
export LOCAL_PREFIX="$LOCAL_PREFIX"

# Prepend local binaries
export PATH="$LOCAL_PREFIX/usr/bin:/usr/bin:$PATH"

# Prepend local libraries
export LD_LIBRARY_PATH="/usr/lib:$LOCAL_PREFIX/usr/lib:$LOCAL_PREFIX/usr/lib/x86_64-linux-gnu:$LOCAL_PREFIX/lib/x86_64-linux-gnu:$LOCAL_PREFIX/usr/lib/gcc/x86_64-linux-gnu/13"

export LIBRARY_PATH="/usr/lib:$LOCAL_PREFIX/usr/lib:$LOCAL_PREFIX/usr/lib/x86_64-linux-gnu:$LOCAL_PREFIX/lib/x86_64-linux-gnu:$LOCAL_PREFIX/usr/lib/gcc/x86_64-linux-gnu/13"

# PKG_CONFIG for local prefix
export PKG_CONFIG_PATH="$LOCAL_PREFIX/usr/lib/pkgconfig:$LOCAL_PREFIX/usr/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH"

# OpenCL vendors in local prefix
# export OPENCL_VENDOR_PATH="$LOCAL_PREFIX/etc/OpenCL/vendors"
# --- End of local build environment setup ---
EOF
        echo "Local environment setup appended to $BASHRC"
    else
        echo "Local environment setup already exists in $BASHRC"
    fi
    
    exec bash
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
    
########################################### ARCH ############################################## 
elif [ -f /etc/arch-release ]; then
    echo "Arch Linux detected."
    check_tools tools_arch missing_tools

    echo "Downloading and installing missing tools locally..."

    for t in "${missing_tools[@]}"; do
        # echo "Downloading and installing $t ..."
            install_arch_package "$t"
    done

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

    grep -qxF "export OCL_ICD_VENDORS="$LOCAL_PREFIX/usr/share/OpenCL/vendors"" ~/.bashrc || \
    echo "export OCL_ICD_VENDORS="$LOCAL_PREFIX/usr/share/OpenCL/vendors"" >> ~/.bashrc

    exec bash
    
    echo "Environment setup complete."
else
    echo "Unsupported Linux distribution. Please run this script on Debian/Ubuntu, Fedora, Arch Linux, or openSUSE."
    exit 1
fi

# Run the program
git config --global core.autocrlf false
chmod +x $SCRIPT_DIR/PIRO_devices_LIN
