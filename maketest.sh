#!/bin/bash
set -eE 
extension=""

show_help() {
    echo "Usage: $0 [-h] [-p PLATFORMID -d DEVICEID]"
    echo
    echo "Example script to demonstrate argument handling in bash."
    echo
    echo "Options:"
    echo "  -h         Display this help message and exit"
    echo "  -p PLATFORMID    Specify the platformid"
    echo "  -d DEVICEID      Specify the deviceid"
}

detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Operating System: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Operating System: macOS"
    elif [[ "$OSTYPE" == "cygwin" ]]; then
        echo "Operating System: Cygwin (POSIX compatibility layer on Windows)"
    elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* || "$OSTYPE" == "win32"* || "$OSTYPE" == "mingw"* ]]; then
        extension=".exe"
        echo "Operating System: MinGW (Minimal GNU for Windows)"
    elif [[ "$OSTYPE" == "win32" ]]; then
        echo "Operating System: Windows"
    elif [[ "$OSTYPE" == "freebsd"* ]]; then
        echo "Operating System: FreeBSD"
    else
        echo "Operating System: Unknown"
    fi
}
# Initialize variables
PLATFORMID=""
DEVICEID=""
FILE=""
LOCAL_PREFIX="$(pwd)/local"
# Parse command line arguments
while getopts ":hp:d:" opt; do
    case ${opt} in
        h)
            show_help
            exit 0
            ;;
        p)
            PLATFORMID=$OPTARG
            ;;
        d)
            DEVICEID=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            show_help
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            show_help
            exit 1
            ;;
    esac
done

# Remove parsed options from positional parameters
shift $((OPTIND - 1))

# Now handle positional arguments (e.g., the `0` at the end)
INDEX=$1

# Check if required arguments are provided
if [ -z "$PLATFORMID" ]; then
    echo "Error: Missing required arguments."
    show_help
    exit 1
fi

# Main script logic
echo "Setup PLATFORMID: $PLATFORMID"
echo "Setup DEVICEID: $DEVICEID"

detect_os
FILE="setup.ini"

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' -e "s/^id = .*/id = $DEVICEID/" \
              -e "s/^platformid = .*/platformid = $PLATFORMID/" \
              ./dependencies/assets/setup.ini
else
    # Linux or Git Bash / WSL
    sed -i -e "s/^id = .*/id = $DEVICEID/" \
           -e "s/^platformid = .*/platformid = $PLATFORMID/" \
           ./dependencies/assets/setup.ini
fi
# compile the program
if [[ -d "$LOCAL_PREFIX" ]]; then
    echo "Rootless setup"
    source ./scripts/linux/env_setup.sh
fi
./makeall.sh
echo "Build : SUCCESSFUL"
# generate the kernels
# cp -r ./dependencies/assets ./bin/.
mkdir -p logs

NEWPATH="/c/cygwin64/bin"
export PATH="$NEWPATH:$PATH"

# run test cases
for file in ./Test/*; do
    if [ -f "$file" ]; then
        echo "Running example : $(basename "$file")"
        fname=$(basename "$file")
        cp "$file" ./bin/assets/$fname
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' -e "s/^id = .*/id = $DEVICEID/" \
                    -e "s/^platformid = .*/platformid = $PLATFORMID/" \
                    ./bin/assets/$fname
        else
            # Linux or Git Bash / WSL
            sed -i -e "s/^id = .*/id = $DEVICEID/" \
                -e "s/^platformid = .*/platformid = $PLATFORMID/" \
                ./bin/assets/$fname
        fi 
        ./bin/PIRO$extension $fname 0 &> ./logs/test_$(basename "$file").txt
    fi
done
echo "Test : RUNS SUCCESSFUL"