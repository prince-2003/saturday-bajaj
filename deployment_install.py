#!/usr/bin/env python3
"""
deployment_install.py - Python script to install and verify dependencies
"""

import subprocess
import sys

def run_command(command):
    """Run a command and return the result."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def main():
    print("🚀 Installing Python dependencies for deployment...")
    
    # Upgrade pip first
    print("📦 Upgrading pip...")
    success, output = run_command("pip install --upgrade pip")
    if not success:
        print(f"❌ Failed to upgrade pip: {output}")
        return False
    
    # Install requirements
    print("📦 Installing requirements...")
    success, output = run_command("pip install -r requirements.txt")
    if not success:
        print(f"❌ Failed to install requirements: {output}")
        return False
    
    # Verify critical imports
    print("🔍 Verifying critical imports...")
    
    critical_packages = [
        "structlog",
        "fastapi", 
        "uvicorn",
        "pydantic_settings",
        "openai",
        "qdrant_client",
        "redis",
        "PyPDF2",
        "pdfplumber",
        "psutil",
        "numpy",
        "aiofiles",
        "httpx",
        "fitz",
        "flashrank"
    ]
    
    all_good = True
    for package in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package} OK")
        except ImportError as e:
            print(f"❌ {package} FAILED: {e}")
            all_good = False
    
    if all_good:
        print("✅ All dependencies verified successfully!")
        return True
    else:
        print("❌ Some dependencies failed verification!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
