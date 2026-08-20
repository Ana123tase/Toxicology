import sys
print('Python:', sys.version)

packages = ['rdkit', 'catboost', 'pandas', 'numpy', 'sklearn', 'tdc']
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'version unknown')
        print(f'{pkg}: OK ({version})')
    except ImportError:
        print(f'{pkg}: NOT INSTALLED')