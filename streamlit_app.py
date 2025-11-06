"""
Streamlit app entry point for Streamlit Cloud deployment.
This file should be in the root directory for Streamlit Cloud to recognize it.
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run the main app
from app import main

if __name__ == "__main__":
    main()

