#!/usr/bin/env bash
export LOCAL_PREFIX="$(pwd)/local"
export PATH="$LOCAL_PREFIX/usr/bin:/usr/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib:$LOCAL_PREFIX/usr/lib:$LOCAL_PREFIX/usr/lib/x86_64-linux-gnu:$LOCAL_PREFIX/lib/x86_64-linux-gnu:$LOCAL_PREFIX/usr/lib/gcc/x86_64-linux-gnu/13"
export LIBRARY_PATH="/usr/lib:$LOCAL_PREFIX/usr/lib:$LOCAL_PREFIX/usr/lib/x86_64-linux-gnu:$LOCAL_PREFIX/lib/x86_64-linux-gnu:$LOCAL_PREFIX/usr/lib/gcc/x86_64-linux-gnu/13"
export PKG_CONFIG_PATH="$LOCAL_PREFIX/usr/lib/pkgconfig:$LOCAL_PREFIX/usr/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH"
export OPENCL_VENDOR_PATH="/etc/OpenCL/vendors"
echo "Environment set up for this shell session."