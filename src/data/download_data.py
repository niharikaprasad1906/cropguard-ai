import os
import subprocess
import sys

def download_dataset(dataset, path):
    os.makedirs(path, exist_ok=True)

    command = [
        sys.executable, "-m", "kaggle.cli",
        "datasets", "download",
        "-d", dataset,
        "-p", path,
        "--unzip"
    ]

    subprocess.run(command)

if __name__ == "__main__":
    print("Downloading Plant Disease Dataset...")
    download_dataset("emmarex/plantdisease", "data/raw/disease")

    print("Downloading Crop Yield Dataset...")
    download_dataset("abhinand05/crop-production-in-india", "data/raw/yield")

    print("Downloading Sensor Dataset...")
    download_dataset("garystafford/environmental-sensor-data-132k", "data/raw/sensor")

    print("All datasets downloaded successfully!")