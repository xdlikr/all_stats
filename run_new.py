#!/usr/bin/env python3
"""
Script to run the new dashboard layout for Process Capability Analysis
"""

import subprocess
import sys
import os

def main():
    # Change to the app directory
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
    
    # Run the new app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app_new.py",
            "--server.port=8501",
            "--server.headless=false",
            "--browser.gatherUsageStats=false"
        ], check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error running app: {e}")

if __name__ == "__main__":
    main()