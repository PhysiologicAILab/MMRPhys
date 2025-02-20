#!/usr/bin/env bash
set -e

echo "Installing Miniconda..."
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh

# Install Miniconda silently
./miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh

# Add conda to path
export PATH="$HOME/miniconda/bin:$PATH"

# Initialize conda for bash
$HOME/miniconda/bin/conda init bash
source ~/.bashrc

echo "Creating conda environment 'dev'..."
conda create -y -n dev python=3.11.2

echo "Activating environment and installing requirements..."
conda activate dev

# Install CUDA toolkit for GPU support
conda install -y cudatoolkit=11.8

# Install packages from requirements.txt
pip install -r requirements.txt

echo "Setup complete! You can now activate the environment with: conda activate dev"