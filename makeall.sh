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