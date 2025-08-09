if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    make -f Makefile.devices clean
    make -f Makefile.devices
    make -f Makefile.devices setup
    chmod +x ./scripts/linux/PIRO_devices_LIN
elif [[ "$OSTYPE" == "msys" ]]; then
    NEWPATH="/c/cygwin64/bin"
    export PATH="$NEWPATH:$PATH"
    make -f Makefile.devices clean
    make -f Makefile.devices
    make -f Makefile.devices setup
fi