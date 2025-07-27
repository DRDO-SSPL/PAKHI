#!/usr/bin/env python3
"""
Installation script for Mini OS for Machine Learning and Training Databases - Monitoring System
Checks dependencies and installs missing packages
"""

import subprocess
import sys
import importlib
import os

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("Mini OS for Machine Learning and Training Databases - Monitoring System - Dependency Checker")
    print("=" * 50)
    
    # Essential packages
    essential_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("psutil", "psutil"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("websockets", "websockets"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("python-dateutil", "dateutil"),
    ]
    
    # Optional packages
    optional_packages = [
        ("GPUtil", "GPUtil"),
        ("tensorflow", "tensorflow"),
        ("scikit-learn", "sklearn"),
        ("plotly", "plotly"),
        ("docker", "docker"),
        ("Pillow", "PIL"),
        ("tqdm", "tqdm"),
        ("colorama", "colorama"),
    ]
    
    missing_essential = []
    missing_optional = []
    
    print("\nChecking essential packages...")
    for package_name, import_name in essential_packages:
        if check_package(package_name, import_name):
            print(f"✓ {package_name}")
        else:
            print(f"✗ {package_name} - MISSING")
            missing_essential.append(package_name)
    
    print("\nChecking optional packages...")
    for package_name, import_name in optional_packages:
        if check_package(package_name, import_name):
            print(f"✓ {package_name}")
        else:
            print(f"✗ {package_name} - MISSING (optional)")
            missing_optional.append(package_name)
    
    # Install missing packages
    if missing_essential:
        print(f"\nInstalling {len(missing_essential)} essential packages...")
        for package in missing_essential:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✓ {package} installed successfully")
            else:
                print(f"✗ Failed to install {package}")
    
    if missing_optional:
        print(f"\nOptional packages missing: {', '.join(missing_optional)}")
        choice = input("Install optional packages? (y/n): ").lower().strip()
        if choice == 'y':
            for package in missing_optional:
                print(f"Installing {package}...")
                if install_package(package):
                    print(f"✓ {package} installed successfully")
                else:
                    print(f"✗ Failed to install {package}")
    
    print("\n" + "=" * 50)
    print("Installation complete!")
    
    # Test imports
    print("\nTesting monitoring system imports...")
    try:
        from training_monitor import training_monitor
        print("✓ Training monitor imported successfully")
    except ImportError as e:
        print(f"✗ Training monitor import failed: {e}")
    
    try:
        from gpu_monitor import gpu_monitor
        print("✓ GPU monitor imported successfully")
    except ImportError as e:
        print(f"✗ GPU monitor import failed: {e}")
    
    try:
        from visualization import MetricsVisualizer
        print("✓ Visualization system imported successfully")
    except ImportError as e:
        print(f"✗ Visualization system import failed: {e}")
    
    print("\nSetup complete! You can now use the monitoring system.")

if __name__ == "__main__":
    main()
