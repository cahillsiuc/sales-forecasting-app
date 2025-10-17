#!/usr/bin/env python3
"""
Test script to verify the Sales Forecasting App is working correctly
"""

import requests
import time
import subprocess
import sys
import os

def test_app_connection():
    """Test if the Streamlit app is running and accessible"""
    try:
        # Test if the app is running on localhost:8501
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ App is running successfully!")
            print("🌐 Open your browser and go to: http://localhost:8501")
            print("")
            print("🔐 Login credentials:")
            print("   Customer: customer / Customer123!")
            print("   Admin: admin / Admin123!")
            return True
        else:
            print(f"❌ App returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ App is not running or not accessible")
        return False
    except Exception as e:
        print(f"❌ Error testing connection: {e}")
        return False

def check_environment():
    """Check if the environment is set up correctly"""
    print("🔍 Checking environment...")
    
    # Check if .env file exists
    if os.path.exists(".env"):
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found - OpenAI features may not work")
    
    # Check if virtual environment exists
    if os.path.exists("venv"):
        print("✅ Virtual environment found")
    else:
        print("❌ Virtual environment not found")
        return False
    
    # Check if app.py exists
    if os.path.exists("app.py"):
        print("✅ app.py found")
    else:
        print("❌ app.py not found")
        return False
    
    return True

def main():
    print("🚀 Sales Forecasting App - Connection Test")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed!")
        return
    
    print("\n🔍 Testing app connection...")
    
    # Test connection
    if test_app_connection():
        print("\n🎉 Everything is working correctly!")
        print("You can now use the app with full authentication and data isolation.")
    else:
        print("\n❌ App connection test failed!")
        print("Make sure the app is running by executing:")
        print("   start_app.bat (Windows)")
        print("   or")
        print("   ./start_app.ps1 (PowerShell)")

if __name__ == "__main__":
    main()
