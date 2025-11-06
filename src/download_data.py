"""
Download dataset from Kaggle using API credentials.
"""
import os
import json
import shutil
from pathlib import Path

def setup_kaggle_credentials():
    """Set up Kaggle API credentials from kaggle.json file."""
    project_root = Path(__file__).parent.parent
    kaggle_json_path = project_root / "kaggle.json"
    
    if not kaggle_json_path.exists():
        raise FileNotFoundError(f"kaggle.json not found at {kaggle_json_path}")
    
    # Read credentials
    with open(kaggle_json_path, 'r') as f:
        credentials = json.load(f)
    
    # Set environment variables for Kaggle API
    os.environ['KAGGLE_USERNAME'] = credentials['username']
    os.environ['KAGGLE_KEY'] = credentials['key']
    
    # Also copy to default Kaggle location for API compatibility
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    default_kaggle_json = kaggle_dir / "kaggle.json"
    
    # Copy credentials file
    shutil.copy2(kaggle_json_path, default_kaggle_json)
    
    # Set proper permissions (important for security on Unix systems)
    if hasattr(os, 'chmod'):
        os.chmod(default_kaggle_json, 0o600)
    
    return credentials

# Set up credentials BEFORE importing Kaggle API
setup_kaggle_credentials()

# Now import Kaggle API (it will use the credentials we just set up)
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    """Download the crime and incarceration dataset from Kaggle."""
    # Set up credentials
    setup_kaggle_credentials()
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Dataset information
    dataset_name = "christophercorrea/prisoners-and-crime-in-united-states"
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    print(f"Downloading dataset: {dataset_name}")
    print(f"Destination: {data_dir}")
    
    # Download dataset
    api.dataset_download_files(
        dataset=dataset_name,
        path=str(data_dir),
        unzip=True
    )
    
    print("Dataset downloaded successfully!")
    print(f"Files saved to: {data_dir}")
    
    # List downloaded files
    files = list(data_dir.glob("*"))
    print("\nDownloaded files:")
    for file in files:
        if file.is_file():
            print(f"  - {file.name}")

if __name__ == "__main__":
    download_dataset()

