#!/bin/bash
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p gpu
#SBATCH --gpus=h100:1
#SBATCH -A bhatele-lab-cmsc 
#SBATCH -J spmm
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --exclusive
#SBATCH -o zaratan_h100.out
# load necessary libraries
source /home/cunyang/torch2.5/bin/activate  
module load cuda

for DIMN in 64 128 256 512 1024; do
#for DIMN in 256 512; do
    # RoDe Sputnik CuSparse
    cd /home/cunyang/scratch.bhatele-lab/FlashSparse
    cd Baseline/RoDe/script/
    python eval_spmm_call_inputN.py $DIMN H100

    # DTC
    cd /home/cunyang/scratch.bhatele-lab/FlashSparse
    cd Baseline/DTC-SpMM/
    python run_DTC_SpMM_h100.py $DIMN H100

    # GNNAdvisor, TCGNN, GeSpmm
    cd /home/cunyang/scratch.bhatele-lab/FlashSparse
    cd eva/kernel/spmm
    python spmm_baselines.py $DIMN H100

    cd /home/cunyang/scratch.bhatele-lab/FlashSparse
    cd eva/kernel/spmm
    python spmm_tf32_test_args.py $DIMN H100
done


