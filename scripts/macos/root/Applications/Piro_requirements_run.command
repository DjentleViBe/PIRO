#!/bin/bash

echo "Installing requirements for running"
echo "🔍 Checking for Git..."

if command -v git >/dev/null 2>&1; then
    echo "✅ Git is already installed:"
    git --version
else
    echo "❌ Git not found."

    if command -v brew >/dev/null 2>&1; then
        echo "🍺 Homebrew found. Installing Git with Homebrew..."
        brew install git
    else
        echo "🔧 Homebrew not found. Trying to install Xcode Command Line Tools..."

        # Prompt for Command Line Tools
        xcode-select --install

        echo ""
        echo "📦 A popup should appear to install the Command Line Tools."
        echo "   Please follow the on-screen instructions."
        echo ""
    fi
    while ! command -v git >/dev/null 2>&1; do
        echo "⏳ Waiting for Git to be installed..."
        sleep 5
    done
    echo "✅ Git is now installed!"
    git --version
fi

git config --global core.autocrlf false
echo ""
cd "$(dirname "$0")"
# Run your executable
./PIRO_devices_MACOS
echo "Please note down the Device and Platform number of the hardware where the simulation is intended to be run."
# Optionally keep Terminal open after execution
read -p "Press [Enter] to close this window..."
