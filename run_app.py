"""
Simple launcher script to help diagnose Flask startup issues
"""
import os
import sys

def check_dependencies():
    """Check if all required packages are installed"""
    required = ["flask", "nltk", "numpy", "scikit-learn"]
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} NOT installed")
    
    if missing:
        print("\nMissing packages. Please install them with:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True

def check_folder_structure():
    """Check if the required folder structure exists"""
    required_folders = [
        "static",
        "static/css",
        "static/js",
        "templates",
        "corpus"
    ]
    
    for folder in required_folders:
        if os.path.exists(folder):
            print(f"✓ Folder '{folder}' exists")
        else:
            print(f"✗ Folder '{folder}' does NOT exist")
            try:
                os.makedirs(folder)
                print(f"  Created folder '{folder}'")
            except Exception as e:
                print(f"  Error creating folder: {e}")

def run_app():
    """Run the Flask application"""
    print("\nStarting Flask application...")
    try:
        from app import app
        app.run(debug=True, host='127.0.0.1', port=5000)
    except Exception as e:
        print(f"Error running Flask app: {e}")
        print("\nDetailed traceback:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Flask AI Application Launcher")
    print("===========================\n")
    
    print("Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\nChecking folder structure...")
    check_folder_structure()
    
    if deps_ok:
        run_app()
    else:
        print("\nPlease install missing dependencies before running the app.")
        print("Run: pip install -r requirements.txt")
