# 🚀 Sales Forecasting App - Startup Guide

## Enhanced Startup Scripts

I've created several robust scripts that eliminate directory and environment issues:

### 🪟 Windows Batch File (Recommended)
**Double-click:** `start_app.bat`

**Enhanced Features:**
- ✅ **Auto-detects script location** - Always runs from correct directory
- ✅ **Verifies all requirements** - Checks for app.py, venv, streamlit
- ✅ **Handles errors gracefully** - Clear error messages and instructions
- ✅ **Activates environment safely** - Proper error handling for activation
- ✅ **Verifies installation** - Confirms streamlit is available before starting
- ✅ **User-friendly output** - Clear status messages and instructions

### 💻 PowerShell Script
**Right-click and "Run with PowerShell":** `start_app.ps1`

**Enhanced Features:**
- ✅ **Colored output** - Easy to read status messages
- ✅ **Exception handling** - Robust error management
- ✅ **Parameter support** - Optional skip checks flag
- ✅ **Detailed logging** - Shows exactly what's happening

### 🎯 Quick Start (From Anywhere)
**Double-click:** `quick_start.bat`

This script can be run from anywhere and will:
- ✅ **Auto-find the project directory**
- ✅ **Navigate to correct location**
- ✅ **Run the main startup script**

### 🧪 Comprehensive Setup Verification
**Run:** `python verify_setup.py`

This will:
- ✅ **Check all essential files** - app.py, requirements.txt, config.yaml
- ✅ **Verify directory structure** - venv, utils, .streamlit
- ✅ **Test Python environment** - Virtual environment, packages
- ✅ **Validate dependencies** - All required packages installed
- ✅ **Check configuration** - .env file with valid API key
- ✅ **Provide detailed feedback** - Clear success/error messages

### 🔍 Simple Connection Test
**Run:** `python test_connection.py`

This will:
- ✅ Check if all files are in place
- ✅ Test if the app is running
- ✅ Verify environment setup

## 📱 Login Screen Preview

I've created a preview of what the login screen looks like:
**Open:** `login_screen_preview.html` in your browser

## 🔐 Login Credentials

**Customer Account:**
- Username: `customer`
- Password: `Customer123!`

**Admin Account:**
- Username: `admin`
- Password: `Admin123!`

## 🎯 What Happens After Login

1. **Customer View:**
   - Upload CSV files (saved to their personal directory)
   - Generate sales forecasts
   - Use AI chat for insights
   - View only their own data

2. **Admin View:**
   - All customer features PLUS
   - Access to admin panel
   - View all user data
   - Manage system settings

## 🛠️ Manual Startup (if scripts don't work)

```bash
# Navigate to project folder
cd sales-forecasting-demo

# Activate virtual environment
venv\Scripts\activate

# Start the app
streamlit run app.py
```

## 🌐 Access the App

Once started, open your browser and go to:
**http://localhost:8501**

## ✅ Verification Checklist

- [ ] Virtual environment exists (`venv` folder)
- [ ] .env file exists with OpenAI API key
- [ ] app.py is present
- [ ] Scripts run without errors
- [ ] Browser opens to login screen
- [ ] Can login with both customer and admin accounts
- [ ] Data isolation works (admin sees all, customer sees own)

## 🆘 Troubleshooting

**If scripts don't work:**
1. Make sure you're in the `sales-forecasting-demo` folder
2. Check that `venv` folder exists
3. Try running commands manually
4. Check Windows execution policy for PowerShell

**If app doesn't start:**
1. Check if port 8501 is available
2. Verify .env file has valid OpenAI API key
3. Check virtual environment is activated

## 🎉 Success!

Once everything is working, you'll see:
- Clean login screen
- Working authentication
- User-specific data directories
- Role-based access control
- All forecasting features

Your app is ready for customer testing! 🚀
