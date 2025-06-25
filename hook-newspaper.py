from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all data files from newspaper
datas = collect_data_files('newspaper')

# Also collect any submodules
hiddenimports = collect_submodules('newspaper')

# Specifically include the resources directory that contains text processing files
try:
    import newspaper
    import os
    newspaper_path = os.path.dirname(newspaper.__file__)
    resources_path = os.path.join(newspaper_path, 'resources')
    if os.path.exists(resources_path):
        # Add the entire resources directory
        datas.append((resources_path, 'newspaper/resources'))
except ImportError:
    pass 