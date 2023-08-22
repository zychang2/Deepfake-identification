#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=CNN

# Load Required Modules
module load python/3.10.4

# pip installs
pip install torch torchvision tqdm pandas numpy scikit-learn Pillow matplotlib


# Set Environment Variables
export DATA_DIR=/home/vsilai/Documents/eecs442_final/data/
export CSV_DIR=/home/vsilai/Documents/eecs442_final/dct_data.csv
export OUTPUT_DIR=/home/vsilai/Documents/eecs442_final/out/

# Run the Code
python CNN.py