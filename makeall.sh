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
    rm -f Executables/Linux.zip
    mkdir -p Tempzip/bin
    rsync -av \
  --include='assets/' \
  --include='assets/***' \
  --include='*.bin' \
  --include='*.exe' \
  --exclude='*' \
  bin/ Tempzip/bin
    rm Tempzip/bin/assets/kernels/*.cl
    zip -r Executables/Linux.zip Tempzip/bin
    rm -r Tempzip
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Creating MacOS.zip"
    rm -f Executables
    mkdir -p Tempzip/bin
    rsync -av \
  --include='assets/' \
  --include='assets/***' \
  --include='*.bin' \
  --include='*.exe' \
  --exclude='*' \
  bin/ Tempzip/bin
    rm Tempzip/bin/assets/kernels/*.cl
    rm Tempzip/assets/kernels/*.cl
    zip -r Executables/MacOS.zip Tempzip/bin
    rm -r Tempzip
elif [[ "$OSTYPE" == "cygwin" ]]; then
    echo "Operating System: Cygwin (POSIX compatibility layer on Windows)"
elif [[ "$OSTYPE" == "msys" ]]; then
    echo "Creating Windows.zip"
    mkdir -p Tempzip/bin
    rsync -av \
  --include='assets/' \
  --include='assets/***' \
  --include='*.bin' \
  --include='*.exe' \
  --exclude='*' \
  bin/ Tempzip/bin
    rm Tempzip/bin/assets/kernels/*.cl
    powershell.exe Compress-Archive -Path '"Tempzip/bin"' -DestinationPath '"Executables\\Windows.zip"' -Force
    rm -r Tempzip
elif [[ "$OSTYPE" == "win32" ]]; then
    echo "Operating System: Windows"
elif [[ "$OSTYPE" == "freebsd"* ]]; then
    echo "Operating System: FreeBSD"
else
    echo "Operating System: Unknown"
fi

