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

echo "Removing conda environment 'mmrphys', if it exists..."
conda remove --name mmrphys --all -y

echo "Creating conda environment 'mmrphys'..."
conda create -y -n mmrphys python=3.11.2 pytorch=2.0.0 torchvision=0.15.1 torchaudio=2.0.1 cudatoolkit=11.8 -c pytorch -q -y

echo "Activating environment and installing requirements..."
conda activate mmrphys

# Install packages from requirements.txt
pip install -r requirements.txt

echo "Setup complete! You can now activate the environment with: conda activate mmrphys"