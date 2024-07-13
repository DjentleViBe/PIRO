extension=""

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

detect_os

cp -r ./dependencies/assets ./bin/.
mkdir -p logs

./bin/GIRO$extension &> ./logs/output.txt