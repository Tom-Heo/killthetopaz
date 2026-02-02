#!/bin/bash

echo "Starting DIV2K dataset download..."

# Create data directory if it doesn't exist
mkdir -p data
cd data

# Define URLs
TRAIN_HR_URL="http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
VALID_HR_URL="http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"

# Function to download and unzip
download_and_extract() {
    local url=$1
    local zip_file=$(basename "$url")
    local folder_name="${zip_file%.zip}"

    if [ -d "$folder_name" ]; then
        echo "Folder $folder_name already exists. Skipping download."
    else
        if [ ! -f "$zip_file" ]; then
            echo "Downloading $zip_file..."
            wget "$url"
        else
            echo "$zip_file already exists. Skipping download."
        fi
        
        echo "Unzipping $zip_file..."
        unzip -q "$zip_file"
        
        # Clean up zip file to save space (optional, commented out)
        # rm "$zip_file"
    fi
}

# Download Training Set
download_and_extract "$TRAIN_HR_URL"

# Download Validation Set
download_and_extract "$VALID_HR_URL"

echo "Dataset preparation complete!"
echo "Structure:"
ls -F
cd ..
