LOCAL_PREFIX="$(pwd)/local"
if [[ -d "$LOCAL_PREFIX" ]]; then
    source ./scripts/linux/env_setup.sh
fi
echo "Please note down the Device and Platform number of the hardware where the simulation is intended to be run."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    ./scripts/linux/PIRO_devices_LIN
fi