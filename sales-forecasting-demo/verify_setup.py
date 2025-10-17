#!/usr/bin/env python3
"""
Comprehensive setup verification script for Sales Forecasting App
This script checks all requirements and provides detailed feedback
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_status(message, status="INFO"):
    """Print status message with color coding"""
    colors = {
        "SUCCESS": "\033[92m",  # Green
        "ERROR": "\033[91m",    # Red
        "WARNING": "\033[93m",  # Yellow
        "INFO": "\033[94m",     # Blue
        "RESET": "\033[0m"      # Reset
    }
    
    status_symbols = {
        "SUCCESS": "✅",
        "ERROR": "❌",
        "WARNING": "⚠️",
        "INFO": "ℹ️"
    }
    
    print(f"{colors.get(status, '')}{status_symbols.get(status, '')} {message}{colors['RESET']}")

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print_status(f"{description}: Found", "SUCCESS")
        return True
    else:
        print_status(f"{description}: Not found", "ERROR")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    if os.path.isdir(dirpath):
        print_status(f"{description}: Found", "SUCCESS")
        return True
    else:
        print_status(f"{description}: Not found", "ERROR")
        return False

def check_python_package(package_name, description):
    """Check if a Python package is installed"""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is not None:
            print_status(f"{description}: Installed", "SUCCESS")
            return True
        else:
            print_status(f"{description}: Not installed", "ERROR")
            return False
    except ImportError:
        print_status(f"{description}: Not installed", "ERROR")
        return False

def check_virtual_environment():
    """Check if we're in a virtual environment"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_status("Virtual environment: Active", "SUCCESS")
        return True
    else:
        print_status("Virtual environment: Not active", "WARNING")
        return False

def check_streamlit_installation():
    """Check if Streamlit is properly installed"""
    try:
        result = subprocess.run([sys.executable, "-c", "import streamlit; print(streamlit.__version__)"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print_status(f"Streamlit: Installed (v{version})", "SUCCESS")
            return True
        else:
            print_status("Streamlit: Installation failed", "ERROR")
            return False
    except Exception as e:
        print_status(f"Streamlit: Check failed - {str(e)}", "ERROR")
        return False

def check_env_file():
    """Check if .env file exists and has required content"""
    if not os.path.exists('.env'):
        print_status(".env file: Not found", "ERROR")
        return False
    
    try:
        with open('.env', 'r') as f:
            content = f.read()
            if 'OPENAI_API_KEY' in content and 'sk-' in content:
                print_status(".env file: Valid OpenAI API key found", "SUCCESS")
                return True
            else:
                print_status(".env file: Missing or invalid OpenAI API key", "WARNING")
                return False
    except Exception as e:
        print_status(f".env file: Error reading - {str(e)}", "ERROR")
        return False

def main():
    """Main verification function"""
    print_header("Sales Forecasting App - Setup Verification")
    print()
    
    # Track overall status
    all_checks_passed = True
    
    # Check current directory
    current_dir = os.getcwd()
    print_status(f"Current directory: {current_dir}", "INFO")
    print()
    
    # Check essential files
    print_header("Essential Files Check")
    essential_files = [
        ("app.py", "Main application file"),
        ("requirements.txt", "Dependencies file"),
        ("config.yaml", "Authentication configuration"),
        (".gitignore", "Git ignore file")
    ]
    
    for filepath, description in essential_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    print()
    
    # Check directories
    print_header("Directory Structure Check")
    essential_dirs = [
        ("venv", "Virtual environment"),
        ("utils", "Utility modules"),
        (".streamlit", "Streamlit configuration")
    ]
    
    for dirpath, description in essential_dirs:
        if not check_directory_exists(dirpath, description):
            all_checks_passed = False
    
    print()
    
    # Check Python environment
    print_header("Python Environment Check")
    print_status(f"Python version: {sys.version}", "INFO")
    
    if not check_virtual_environment():
        all_checks_passed = False
    
    print()
    
    # Check required packages
    print_header("Required Packages Check")
    required_packages = [
        ("streamlit", "Streamlit web framework"),
        ("pandas", "Data manipulation"),
        ("plotly", "Interactive charts"),
        ("scikit-learn", "Machine learning"),
        ("openai", "OpenAI API client"),
        ("python-dotenv", "Environment variables")
    ]
    
    for package, description in required_packages:
        if not check_python_package(package, description):
            all_checks_passed = False
    
    print()
    
    # Check Streamlit specifically
    print_header("Streamlit Installation Check")
    if not check_streamlit_installation():
        all_checks_passed = False
    
    print()
    
    # Check environment configuration
    print_header("Environment Configuration Check")
    if not check_env_file():
        all_checks_passed = False
    
    print()
    
    # Final summary
    print_header("Verification Summary")
    if all_checks_passed:
        print_status("All checks passed! Your setup is ready.", "SUCCESS")
        print()
        print_status("You can now start the app using:", "INFO")
        print("  • Double-click: start_app.bat")
        print("  • PowerShell: .\\start_app.ps1")
        print("  • Command line: streamlit run app.py")
        print()
        print_status("The app will be available at: http://localhost:8501", "INFO")
    else:
        print_status("Some checks failed. Please fix the issues above.", "ERROR")
        print()
        print_status("Common fixes:", "INFO")
        print("  • Create virtual environment: python -m venv venv")
        print("  • Activate virtual environment: venv\\Scripts\\activate")
        print("  • Install dependencies: pip install -r requirements.txt")
        print("  • Create .env file with your OpenAI API key")
    
    print()
    return all_checks_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nVerification cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        sys.exit(1)
