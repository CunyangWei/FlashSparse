#!/bin/sh
#for DIMN in 64 128 256 512 1024; do
for DIMN in 128 256 512 1024; do
    # RoDe Sputnik CuSparse
    #cd /home/cunyang/scratch.bhatele-lab/FlashSparse
    #cd Baseline/RoDe/script/
    #python eval_spmm_call_inputN.py $DIMN A100

    # DTC
    #cd /home/cunyang/scratch.bhatele-lab/FlashSparse
    #cd Baseline/DTC-SpMM/
    #python run_DTC_SpMM_h100.py $DIMN A100

    # GNNAdvisor, TCGNN, GeSpmm
    #cd /home/cunyang/scratch.bhatele-lab/FlashSparse
    #cd eva/kernel/spmm
    #python spmm_baselines.py $DIMN A100

    cd /home/cunyang/scratch.bhatele-lab/FlashSparse
    cd eva/kernel/spmm
    python spmm_tf32_test_args.py $DIMN A100
done
