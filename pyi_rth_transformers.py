import os
import sys
import warnings

# Runtime hook for transformers library to fix PyInstaller compatibility
def patch_transformers_import():
    """Patch transformers import structure creation to work with PyInstaller"""
    try:
        import transformers.utils.import_utils
        
        # Store the original function
        original_create_import_structure = transformers.utils.import_utils.create_import_structure_from_path
        
        def patched_create_import_structure_from_path(module_path):
            """Patched version that handles PyInstaller's file structure"""
            try:
                # If the path ends with .pyc, try the .py version
                if module_path.endswith('.pyc'):
                    py_path = module_path[:-1]  # Remove 'c' from '.pyc'
                    if os.path.exists(py_path):
                        module_path = py_path
                    else:
                        # Try without the __init__.pyc part
                        base_path = os.path.dirname(module_path)
                        if os.path.exists(base_path):
                            module_path = base_path
                        else:
                            # If we can't find the path, return empty structure silently
                            return {}
                
                return original_create_import_structure(module_path)
            except (FileNotFoundError, OSError) as e:
                # Silently return empty structure instead of printing warnings
                return {}
        
        # Apply the patch
        transformers.utils.import_utils.create_import_structure_from_path = patched_create_import_structure_from_path
        
    except ImportError:
        # transformers not available, skip patching
        pass

def suppress_transformers_warnings():
    """Suppress the noisy transformers warnings about missing .pyc files"""
    try:
        # Redirect stderr temporarily to suppress the 🚨 warnings
        import transformers.utils.logging
        
        # Set transformers logging to ERROR level to reduce noise
        transformers.utils.logging.set_verbosity_error()
        
        # Also try to suppress the specific warnings about model paths
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
    except ImportError:
        pass

# Apply the patches when this hook is loaded
patch_transformers_import()
suppress_transformers_warnings() 