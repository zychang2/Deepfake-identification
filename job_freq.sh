#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=install_cv2

# Load Required Modules
module load python/3.10.4

# pip installs
pip install torch torchvision tqdm pandas numpy scikit-learn Pillow matplotlib opencv-python


# Set Environment Variables
export DATA_DIR=/home/vsilai/Documents/eecs442_final/data/
export CSV_DIR=/home/vsilai/Documents/eecs442_final/data.csv
export OUTPUT_DIR=/home/vsilai/Documents/eecs442_final/out/


python preprocess_freq.py
