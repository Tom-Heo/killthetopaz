#!/bin/bash

echo "Starting environment setup..."

# 1. Update Package List
sudo apt-get update

# 2. Install Git (if not installed)
if ! command -v git &> /dev/null
then
    echo "Git could not be found, installing..."
    sudo apt-get install -y git
fi

# 3. Install Python Dependencies
echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# 4. Create Directories
echo "Creating project directories..."
mkdir -p data
mkdir -p checkpoints
mkdir -p results

echo "Environment setup complete!"
