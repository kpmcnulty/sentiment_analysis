from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all data files from nltk
datas = collect_data_files('nltk')

# Also collect any submodules
hiddenimports = collect_submodules('nltk') 