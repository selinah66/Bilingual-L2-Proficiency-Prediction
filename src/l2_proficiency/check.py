import sys
from importlib.metadata import version, PackageNotFoundError

print(f"Python: {sys.version.split()[0]}")

packages = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "openpyxl",
    "Pillow",
    "graphviz",
]

for package in packages:
    try:
        print(f"{package}=={version(package)}")
    except PackageNotFoundError:
        print(f"{package}: not installed")