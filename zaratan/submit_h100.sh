#!/bin/bash
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p gpu
#SBATCH --gpus=h100:1
#SBATCH -A bhatele-lab-cmsc 
#SBATCH -J spmm
#SBATCH --mem=32G
#SBATCH -t 00:40:00
#SBATCH --exclusive
#SBATCH -o spmm_h100.out

source /home/cunyang/spmm/spmm-venv/bin/activate
module load cuda

DIMN=128

cd /home/cunyang/scratch.bhatele-lab/FlashSparse
cd eva/kernel/spmm
python spmm_tf32_test_args.py DIMN


cd /home/cunyang/scratch.bhatele-lab/FlashSparse
cd eva/kernel/spmm
python spmm_baseline.py DIMN

cd /home/cunyang/scratch.bhatele-lab/FlashSparse
cd Baseline/DTC-SpMM
python run_DTC_SpMM_h100.py DIMN

cd /home/cunyang/scratch.bhatele-lab/FlashSparse
cd eva/kernel/spmm
python spmm_tf32_test_args.py DIMN

cd /home/cunyang/scratch.bhatele-lab/FlashSparse
cd Baseline/RoDe/script/
python eval_spmm_call_inputN.py DIMN


