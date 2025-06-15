#./makelib.sh
# Only add Cygwin to PATH if in Git Bash on Windows
if [[ "$MSYSTEM" == "MINGW64" ]] && [[ -d "/c/cygwin64/bin" ]]; then
    if ! grep -q "cygwin64" ~/.bashrc 2>/dev/null; then
        echo 'export PATH="/c/cygwin64/bin:$PATH"' >> ~/.bashrc
        echo "Added Cygwin to ~/.bashrc"
        source ~/.bashrc
    fi
fi
NEWPATH="/c/cygwin64/bin"
export PATH="$NEWPATH:$PATH"
make clean
make
make setup
./bin/PIRO$extension setup.ini 1 &> ./logs/test_buildkernel.txt
echo "Run : KERNEL SUCCESSFUL"

mkdir -p Executables
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Creating Linux.zip"
    rm -f Linux.zip
    zip -r Executables/Linux.zip bin -x '*.cl'
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Creating MacOS.zip"
    rm -f MacOS.zip
    zip -r Executables/MacOS.zip bin -x '*.cl'
elif [[ "$OSTYPE" == "cygwin" ]]; then
    echo "Operating System: Cygwin (POSIX compatibility layer on Windows)"
elif [[ "$OSTYPE" == "msys" ]]; then
    echo "Creating Windows.zip"
    mkdir -p Tempzip
    rsync -av --exclude='*.cl' bin/ Tempzip/
    powershell.exe Compress-Archive -Path '"Tempzip"' -DestinationPath '"Executables\\Windows.zip"' -Force
elif [[ "$OSTYPE" == "win32" ]]; then
    echo "Operating System: Windows"
elif [[ "$OSTYPE" == "freebsd"* ]]; then
    echo "Operating System: FreeBSD"
else
    echo "Operating System: Unknown"
fi

