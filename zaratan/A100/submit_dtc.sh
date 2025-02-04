#!/bin/bash
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH -A bhatele-lab-cmsc 
#SBATCH -J spmm
#SBATCH --mem=64G
#SBATCH -t 00:10:00
#SBATCH --exclusive
#SBATCH -o dtc.out
# load necessary libraries
source /home/cunyang/spmm/spmm-venv/bin/activate  
module load cuda
cd /home/cunyang/scratch.bhatele-lab/FlashSparse
cd Baseline/DTC-SpMM
python run_DTC_SpMM_h100.py 64
