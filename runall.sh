#!/bin/bash
extension=""

show_help() {
    echo "Usage: $0 [-h] [-f NAME]"
    echo
    echo "Example script to demonstrate argument handling in bash."
    echo
    echo "Options:"
    echo "  -h         Display this help message and exit"
    echo "  -f FILE    Specify the name"
}

detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Operating System: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Operating System: macOS"
    elif [[ "$OSTYPE" == "cygwin" ]]; then
        echo "Operating System: Cygwin (POSIX compatibility layer on Windows)"
    elif [[ "$OSTYPE" == "msys" ]]; then
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
FILE=""

# Parse command line arguments
while getopts ":hf:" opt; do
    case ${opt} in
        h)
            show_help
            exit 0
            ;;
        f)
            FILE=$OPTARG
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

# Check if required arguments are provided
if [ -z "$FILE" ]; then
    echo "Error: Missing required arguments."
    show_help
    exit 1
fi

# Main script logic
echo "Setup file: $FILE"

detect_os

cp -r ./dependencies/assets ./bin/.
mkdir -p logs

./bin/GIRO$extension $FILE &> ./logs/output.txt