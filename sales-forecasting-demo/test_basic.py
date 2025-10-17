#!/usr/bin/env python3
"""
AI Sales Forecasting Demo - Basic Test Suite
============================================

Basic test script that validates core functionality without external dependencies.
Tests file structure, data availability, and basic functionality.

Usage:
    python test_basic.py

Author: AI Sales Forecasting Demo Team
"""

import sys
import os
import csv
from datetime import datetime

def test_file_structure():
    """Test that all required files exist"""
    print("🔍 Testing File Structure...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        '.env.template',
        'README.md',
        'test_app.py',
        'utils/__init__.py',
        'utils/data_processing.py',
        'utils/forecasting.py',
        'utils/ai_chat.py',
        'sample_data/sample_sales.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_sample_data():
    """Test sample data structure and content"""
    print("\n📊 Testing Sample Data...")
    
    try:
        # Check if sample data exists
        if not os.path.exists('sample_data/sample_sales.csv'):
            print("❌ Sample data file not found")
            return False
        
        # Read and analyze sample data
        with open('sample_data/sample_sales.csv', 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        if not data:
            print("❌ Sample data is empty")
            return False
        
        # Check required columns
        required_columns = ['ItemID', 'Description', 'TrendType', 'ds', 'CustomerName', 'y']
        actual_columns = list(data[0].keys())
        
        missing_columns = [col for col in required_columns if col not in actual_columns]
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        
        # Analyze data content
        total_records = len(data)
        unique_items = len(set(row['ItemID'] for row in data))
        unique_customers = len(set(row['CustomerName'] for row in data))
        unique_dates = len(set(row['ds'] for row in data))
        
        print(f"✅ Sample data loaded successfully")
        print(f"   - Total records: {total_records:,}")
        print(f"   - Unique items: {unique_items}")
        print(f"   - Unique customers: {unique_customers}")
        print(f"   - Unique dates: {unique_dates}")
        
        # Check data quality
        trend_types = set(row['TrendType'] for row in data)
        if 'Regular' not in trend_types or 'Irregular' not in trend_types:
            print("❌ Missing trend types")
            return False
        
        print(f"✅ Data quality checks passed")
        print(f"   - Trend types: {sorted(trend_types)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading sample data: {str(e)}")
        return False

def test_app_structure():
    """Test main app.py structure"""
    print("\n📱 Testing App Structure...")
    
    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key components
        required_components = [
            'import streamlit as st',
            'def main():',
            'if __name__ == "__main__":',
            'render_header',
            'render_file_upload_section',
            'render_data_validation_section',
            'render_forecasting_dashboard',
            'DEMO_CONFIG',
            'show_help_section'
        ]
        
        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)
        
        if missing_components:
            print(f"❌ Missing components: {missing_components}")
            return False
        
        print("✅ App structure validation passed")
        
        # Check for demo features
        demo_features = [
            'Demo Mode',
            'demo_mode',
            'DEMO_CONFIG',
            'render_demo_tips',
            'show_help_section'
        ]
        
        demo_found = sum(1 for feature in demo_features if feature in content)
        print(f"✅ Demo features found: {demo_found}/{len(demo_features)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading app.py: {str(e)}")
        return False

def test_requirements():
    """Test requirements.txt structure"""
    print("\n📦 Testing Requirements...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        required_packages = [
            'streamlit',
            'pandas',
            'prophet',
            'plotly',
            'openai',
            'python-dotenv',
            'scikit-learn',
            'numpy'
        ]
        
        missing_packages = []
        for package in required_packages:
            if not any(package in req for req in requirements):
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {missing_packages}")
            return False
        
        print("✅ All required packages found in requirements.txt")
        print(f"   - Total packages: {len(requirements)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {str(e)}")
        return False

def test_documentation():
    """Test documentation completeness"""
    print("\n📚 Testing Documentation...")
    
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        required_sections = [
            'Quick Start',
            'Demo Script',
            'Troubleshooting',
            'Feature Overview',
            'Installation',
            'Run the application'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in readme_content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ Missing documentation sections: {missing_sections}")
            return False
        
        print("✅ Documentation completeness check passed")
        print(f"   - README.md size: {len(readme_content):,} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading README.md: {str(e)}")
        return False

def run_all_tests():
    """Run all basic tests"""
    print("🚀 Starting AI Sales Forecasting Demo - Basic Test Suite")
    print("=" * 65)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Sample Data", test_sample_data),
        ("App Structure", test_app_structure),
        ("Requirements", test_requirements),
        ("Documentation", test_documentation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Critical error in {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Generate summary
    print("\n" + "=" * 65)
    print("📊 TEST SUMMARY")
    print("=" * 65)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        print("\n❌ FAILED TESTS:")
        for test_name, result in results:
            if not result:
                print(f"  • {test_name}")
    
    print("\n" + "=" * 65)
    
    if failed_tests == 0:
        print("🎉 ALL BASIC TESTS PASSED!")
        print("📋 Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Run full test suite: python test_app.py")
        print("   3. Start the app: streamlit run app.py")
        return True
    else:
        print("⚠️ Some tests failed. Please review before proceeding.")
        return False

def main():
    """Main test function"""
    print("AI Sales Forecasting Demo - Basic Test Suite")
    print("============================================")
    
    success = run_all_tests()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
