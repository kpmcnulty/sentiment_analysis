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

# Check if we're in the right place (SentimentAnalysisTool folder should be here)
if [ ! -d "$SCRIPT_DIR/SentimentAnalysisTool.app" ] && [ ! -f "$SCRIPT_DIR/SentimentAnalysisTool" ]; then
    echo "Error: Cannot find SentimentAnalysisTool in current directory."
    echo "Please run this installer from inside the unzipped folder."
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

# Remove quarantine attributes from current folder
echo ""
echo "Removing security restrictions (requires password)..."
sudo xattr -cr "$SCRIPT_DIR"

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

# Copy entire folder to Applications (preserving structure)
echo "Installing to Applications folder..."
sudo cp -R "$SCRIPT_DIR" "/Applications/SentimentAnalysisTool"

# Find the actual app path
if [ -f "/Applications/SentimentAnalysisTool/SentimentAnalysisTool.app/Contents/MacOS/SentimentAnalysisTool" ]; then
    APP_PATH="/Applications/SentimentAnalysisTool/SentimentAnalysisTool.app"
elif [ -f "/Applications/SentimentAnalysisTool/SentimentAnalysisTool" ]; then
    # Create a simple launcher script if no .app bundle exists
    echo "Creating app launcher..."
    sudo tee "/Applications/SentimentAnalysisTool/Launch Sentiment Analysis Tool.command" > /dev/null <<EOL
#!/bin/bash
cd "/Applications/SentimentAnalysisTool"
./SentimentAnalysisTool
EOL
    sudo chmod +x "/Applications/SentimentAnalysisTool/Launch Sentiment Analysis Tool.command"
    APP_PATH="/Applications/SentimentAnalysisTool/Launch Sentiment Analysis Tool.command"
else
    echo "Warning: Could not find executable. You may need to launch manually."
    APP_PATH="/Applications/SentimentAnalysisTool"
fi

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
        make alias file to POSIX file "$APP_PATH" at POSIX file "$DESKTOP"
        set name of result to "Sentiment Analysis Tool"
    on error
        -- If alias fails, create a symbolic link instead
        do shell script "ln -s '$APP_PATH' '$DESKTOP/Sentiment Analysis Tool'"
    end try
end tell
EOD

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