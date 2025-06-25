import os
import sys

# Runtime hook for NLTK to set correct data path in PyInstaller executable
def setup_nltk_data_path():
    """Set up NLTK data path for PyInstaller executable"""
    try:
        import nltk
        
        # Get the executable directory
        if getattr(sys, 'frozen', False):
            # Running in PyInstaller bundle
            bundle_dir = sys._MEIPASS
            nltk_data_path = os.path.join(bundle_dir, 'nltk_data')
            
            # Add the bundled NLTK data path to the beginning of the search path
            if os.path.exists(nltk_data_path):
                nltk.data.path.insert(0, nltk_data_path)
        
    except ImportError:
        # NLTK not available, skip setup
        pass

# Apply the setup when this hook is loaded
setup_nltk_data_path() 