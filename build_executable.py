import os
import sys
import subprocess
import shutil
import platform
import json

def check_pyinstaller():
    """Check if PyInstaller is installed, install if not"""
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def create_macos_app_bundle():
    """Create a minimal macOS .app wrapper that points to the onedir executable"""
    print("Creating minimal macOS .app wrapper...")
    
    # Paths
    onedir_path = "dist/SentimentAnalysisTool"
    app_bundle_path = "dist/SentimentAnalysisTool.app"
    
    if not os.path.exists(onedir_path):
        print(f"Error: {onedir_path} not found")
        return False
    
    # Remove existing app bundle if it exists
    if os.path.exists(app_bundle_path):
        shutil.rmtree(app_bundle_path)
    
    # Create minimal .app bundle structure
    contents_dir = os.path.join(app_bundle_path, "Contents")
    macos_dir = os.path.join(contents_dir, "MacOS")
    
    os.makedirs(macos_dir, exist_ok=True)
    
    # Create a launcher script that runs the actual executable
    launcher_script = f"""#!/bin/bash
# Get the directory where this .app is located
APP_DIR="$(dirname "$(dirname "$(dirname "$0")")")"
# The executable is in the same directory as the .app
EXECUTABLE_DIR="$APP_DIR"

# Set up environment
export DYLD_LIBRARY_PATH="$EXECUTABLE_DIR/_internal:$DYLD_LIBRARY_PATH"
export PYTHONPATH="$EXECUTABLE_DIR/_internal:$PYTHONPATH"

# Change to the executable directory and run
cd "$EXECUTABLE_DIR"

# The actual executable is in the parent directory of the .app
REAL_EXECUTABLE="$EXECUTABLE_DIR/SentimentAnalysisTool"

# Launch the executable and keep the app bundle active
"$REAL_EXECUTABLE" &
APP_PID=$!

# Wait for the app to finish
wait $APP_PID
"""
    
    launcher_path = os.path.join(macos_dir, "SentimentAnalysisTool")
    with open(launcher_path, "w") as f:
        f.write(launcher_script)
    
    # Make launcher executable
    os.chmod(launcher_path, 0o755)
    
    # Create Info.plist
    info_plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>Sentiment Analysis Tool</string>
    <key>CFBundleExecutable</key>
    <string>SentimentAnalysisTool</string>
    <key>CFBundleIdentifier</key>
    <string>com.local.sentimentanalysistool</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>SentimentAnalysisTool</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSAppleScriptEnabled</key>
    <false/>
    <key>LSBackgroundOnly</key>
    <false/>
</dict>
</plist>"""
    
    # Write Info.plist
    with open(os.path.join(contents_dir, "Info.plist"), "w") as f:
        f.write(info_plist)
    
    print(f"✅ Minimal macOS app wrapper created: {app_bundle_path}")
    
    # Move the .app bundle inside the SentimentAnalysisTool folder
    final_app_path = os.path.join(onedir_path, "SentimentAnalysisTool.app")
    if os.path.exists(final_app_path):
        shutil.rmtree(final_app_path)
    shutil.move(app_bundle_path, final_app_path)
    
    print(f"✅ Moved app bundle to: {final_app_path}")
    print("   Users can double-click the .app, everything stays together!")
    
    return True

def create_macos_alias():
    """Create a simple alias to the executable"""
    print("Creating macOS alias...")
    
    onedir_path = "dist/SentimentAnalysisTool"
    executable_path = os.path.join(onedir_path, "SentimentAnalysisTool")
    alias_path = "dist/Sentiment Analysis Tool"
    
    if not os.path.exists(executable_path):
        print(f"Error: {executable_path} not found")
        return False
    
    # Remove existing alias
    if os.path.exists(alias_path):
        os.remove(alias_path)
    
    # Create alias using AppleScript
    script = f'''
tell application "Finder"
    make alias file to POSIX file "{os.path.abspath(executable_path)}" at POSIX file "{os.path.dirname(os.path.abspath(alias_path))}"
    set name of result to "{os.path.basename(alias_path)}"
end tell
'''
    
    try:
        subprocess.run(['osascript', '-e', script], check=True)
        print(f"✅ Alias created: {alias_path}")
        print("   Users can double-click the alias to run the app!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create alias")
        return False

def build_executable():
    """Build the executable using PyInstaller with proper ML library handling"""
    
    # Check PyInstaller
    check_pyinstaller()
    
    # Create necessary directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)
    
    # Create config files if they don't exist
    scraper_config = {
        "max_articles": 100,
        "timeout": 30,
        "user_agent": "Mozilla/5.0 (compatible; SentimentAnalyzer/1.0)"
    }
    
    sentiment_config = {
        "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "batch_size": 32,
        "max_length": 512
    }
    
    if not os.path.exists("scraper_config.json"):
        with open("scraper_config.json", "w") as f:
            json.dump(scraper_config, f, indent=2)
    
    if not os.path.exists("sentiment_config.json"):
        with open("sentiment_config.json", "w") as f:
            json.dump(sentiment_config, f, indent=2)
    
    # Build the PyInstaller command with comprehensive ML library support
    cmd = [
        'pyinstaller',
        '--onedir',
        '--windowed',
        '--name=SentimentAnalysisTool',
        '--distpath=dist',
        '--workpath=build',
        '--specpath=.',
        
        # Add custom hooks directory
        '--additional-hooks-dir=.',
        
        # Collect all necessary packages
        '--collect-all=transformers',
        '--collect-all=tokenizers',
        '--collect-all=huggingface_hub',
        '--collect-all=sentence_transformers',
        '--collect-all=bertopic',
        '--collect-all=torch',
        '--collect-all=torchvision',
        '--collect-all=torchaudio',
        '--collect-all=sklearn',
        '--collect-all=scipy',
        '--collect-all=numpy',
        '--collect-all=pandas',
        '--collect-all=nltk',
        '--collect-all=spacy',
        '--collect-all=umap',
        '--collect-all=hdbscan',
        '--collect-all=plotly',
        '--collect-all=requests',
        '--collect-all=urllib3',
        '--collect-all=newspaper',
        '--collect-all=beautifulsoup4',
        '--collect-all=lxml',
        '--collect-all=dateparser',
        '--collect-all=GoogleNews',
        
        # Collect newspaper resources specifically
        '--collect-data=newspaper',
        
        # Collect NLTK data specifically
        '--collect-data=nltk_data',
        f'--add-data={os.path.expanduser("~/nltk_data")}:nltk_data',
        
        # Transformers specific hidden imports
        '--hidden-import=transformers.models',
        '--hidden-import=transformers.models.auto',
        '--hidden-import=transformers.models.auto.modeling_auto',
        '--hidden-import=transformers.models.auto.tokenization_auto',
        '--hidden-import=transformers.models.auto.configuration_auto',
        '--hidden-import=transformers.models.roberta',
        '--hidden-import=transformers.models.roberta.modeling_roberta',
        '--hidden-import=transformers.models.roberta.tokenization_roberta',
        '--hidden-import=transformers.models.roberta.tokenization_roberta_fast',
        '--hidden-import=transformers.models.roberta.configuration_roberta',
        '--hidden-import=transformers.models.bert',
        '--hidden-import=transformers.models.bert.modeling_bert',
        '--hidden-import=transformers.models.bert.tokenization_bert',
        '--hidden-import=transformers.models.bert.configuration_bert',
        '--hidden-import=transformers.utils.import_utils',
        '--hidden-import=transformers.file_utils',
        '--hidden-import=transformers.modeling_utils',
        '--hidden-import=transformers.tokenization_utils',
        '--hidden-import=transformers.tokenization_utils_base',
        '--hidden-import=transformers.configuration_utils',
        '--hidden-import=transformers.generation_utils',
        '--hidden-import=transformers.trainer_utils',
        
        # Exclude unused transformers models to reduce warnings
        '--exclude-module=transformers.models.speecht5',
        '--exclude-module=transformers.models.vits',
        '--exclude-module=transformers.models.seamless_m4t',
        '--exclude-module=transformers.models.seamless_m4t_v2',
        '--exclude-module=transformers.models.fastspeech2_conformer',
        
        # Sentence transformers
        '--hidden-import=sentence_transformers.models',
        '--hidden-import=sentence_transformers.models.Transformer',
        '--hidden-import=sentence_transformers.models.Pooling',
        '--hidden-import=sentence_transformers.models.Normalize',
        
        # PyTorch
        '--hidden-import=torch.utils.data',
        '--hidden-import=torch.nn.functional',
        '--hidden-import=torch.optim',
        '--hidden-import=torch.cuda',
        
        # Scikit-learn
        '--hidden-import=sklearn.utils._cython_blas',
        '--hidden-import=sklearn.neighbors.typedefs',
        '--hidden-import=sklearn.neighbors.quad_tree',
        '--hidden-import=sklearn.tree._utils',
        
        # GUI libraries
        '--hidden-import=tkinter',
        '--hidden-import=tkinter.ttk',
        '--hidden-import=tkinter.messagebox',
        '--hidden-import=tkinter.filedialog',
        '--hidden-import=tkinter.scrolledtext',
        
        # Web scraping
        '--hidden-import=requests.packages.urllib3',
        '--hidden-import=urllib3.util.retry',
        '--hidden-import=urllib3.util.connection',
        '--hidden-import=bs4',
        '--hidden-import=lxml.etree',
        '--hidden-import=lxml.html',
        
        # Data processing
        '--hidden-import=pandas.plotting',
        '--hidden-import=pandas.io.formats.excel',
        '--hidden-import=numpy.random.common',
        '--hidden-import=numpy.random.bounded_integers',
        '--hidden-import=numpy.random.entropy',
        
        # Add configuration files
        '--add-data=scraper_config.json:.',
        '--add-data=sentiment_config.json:.',
        
        # Add runtime hook for transformers
        '--runtime-hook=pyi_rth_transformers.py',
        
        # Add runtime hook for NLTK
        '--runtime-hook=pyi_rth_nltk.py',
        
        # Add runtime hook for SSL certificates
        '--runtime-hook=pyi_rth_ssl.py',
        
        # Entry point
        'sentiment_gui.py'
    ]
    
    print("Building executable with PyInstaller...")
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Build completed successfully!")
        print(f"Application bundle created in: dist/SentimentAnalysisTool/")
        
        # Check if macOS app bundle was created
        if platform.system() == "Darwin":
            create_macos_app_bundle()
        
        print("\nNote: The first run may take longer as it downloads necessary models and NLTK data.")
        print("The executable bundle is quite large due to the ML libraries, but this is normal.")
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        print("\nTroubleshooting tips:")
        print("1. Make sure all dependencies are installed in your current environment")
        print("2. Try upgrading PyInstaller: pip install --upgrade pyinstaller")
        print("3. Check if there are any missing hidden imports in the error messages")
        return False
    
    return True

if __name__ == "__main__":
    success = build_executable()
    if success:
        print("\nBuild completed successfully!")
    else:
        print("\n❌ Build failed!")
        sys.exit(1) 