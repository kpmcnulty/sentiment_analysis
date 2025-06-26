#!/bin/bash

# Sentiment Analysis Tool - macOS Installer
# This script should be run from inside the unzipped folder

echo "=================================="
echo "Sentiment Analysis Tool Installer"
echo "=================================="
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: This installer is for macOS only."
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if SentimentAnalysisTool folder exists next to this script
if [ ! -d "$SCRIPT_DIR/SentimentAnalysisTool" ]; then
    echo "Error: Cannot find SentimentAnalysisTool folder."
    echo "Please run this installer from the unzipped folder."
    exit 1
fi

echo "This installer will:"
echo "1. Remove security quarantine flags"
echo "2. Move app to your Applications folder"
echo "3. Create a desktop shortcut"
echo ""
echo "You may be prompted for your password."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

# Remove quarantine attributes from the SentimentAnalysisTool folder
echo ""
echo "Removing security restrictions (requires password)..."
sudo xattr -cr "$SCRIPT_DIR/SentimentAnalysisTool"

# Check if app already exists in Applications
if [ -d "/Applications/SentimentAnalysisTool" ]; then
    echo ""
    read -p "App already exists in Applications. Overwrite? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing old version..."
        sudo rm -rf "/Applications/SentimentAnalysisTool"
    else
        echo "Installation cancelled."
        exit 0
    fi
fi

# Move the SentimentAnalysisTool folder to Applications
echo "Installing to Applications folder..."
sudo mv "$SCRIPT_DIR/SentimentAnalysisTool" "/Applications/SentimentAnalysisTool"

# The app path is always the .app inside the folder
APP_PATH="/Applications/SentimentAnalysisTool/SentimentAnalysisTool.app"

# Create desktop alias
echo "Creating desktop shortcut..."
DESKTOP="$HOME/Desktop"

# Remove old alias if exists
if [ -f "$DESKTOP/Sentiment Analysis Tool" ] || [ -L "$DESKTOP/Sentiment Analysis Tool" ]; then
    rm "$DESKTOP/Sentiment Analysis Tool"
fi

# Create alias using osascript (AppleScript)
osascript <<EOD
tell application "Finder"
    try
        set appFile to POSIX file "$APP_PATH" as alias
        make alias file to appFile at desktop
        set name of result to "Sentiment Analysis Tool"
    on error errorMessage
        display dialog "Could not create desktop shortcut: " & errorMessage
    end try
end tell
EOD

# Verify alias was created
if [ ! -e "$DESKTOP/Sentiment Analysis Tool" ]; then
    echo "Note: Desktop shortcut creation failed. You can manually:"
    echo "1. Open Applications folder"
    echo "2. Find SentimentAnalysisTool folder"
    echo "3. Right-click SentimentAnalysisTool.app"
    echo "4. Select 'Make Alias'"
    echo "5. Drag alias to Desktop"
fi

echo ""
echo "=================================="
echo "Installation Complete!"
echo "=================================="
echo ""
echo "✅ App installed to: /Applications/SentimentAnalysisTool/"
echo "✅ Desktop shortcut created: Sentiment Analysis Tool"
echo ""
echo "To run the app:"
echo "• Double-click 'Sentiment Analysis Tool' on your desktop"
echo "• Or open from Applications → SentimentAnalysisTool folder"
echo ""
echo "First run tip: If macOS still complains, right-click"
echo "the app and select 'Open' instead of double-clicking."
echo ""
echo "Enjoy using Sentiment Analysis Tool!"