#!/bin/bash

echo "Downloading DIV2K dataset..."

# Create data directory if it doesn't exist
mkdir -p data
cd data

# Note: DIV2K download links often change or require registration.
# Ideally, you should download DIV2K_train_HR.zip and DIV2K_valid_HR.zip manually
# and place them in the 'data' folder.

# Attempting to download from official source (might fail if links are stale)
# Replace these with valid mirrors if necessary.

echo "Please manually download DIV2K_train_HR.zip and DIV2K_valid_HR.zip"
echo "and unzip them into the 'data' directory."
echo "Structure should be:"
echo "data/"
echo "  DIV2K_train_HR/"
echo "    0001.png"
echo "    ..."
echo "  DIV2K_valid_HR/"
echo "    0801.png"
echo "    ..."

# Example commands if you have the links:
# wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
# wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
# unzip DIV2K_train_HR.zip
# unzip DIV2K_valid_HR.zip

cd ..
echo "Download script finished (check instructions above)."
