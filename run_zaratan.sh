#!/bin/bash


# RoDe Sputnik CuSparse
cd Baseline/RoDe/script/
python eval_spmm_call_inputN.py 64

# DTC
cd Baseline/DTC-SpMM/
python run_DTC_SpMM_h100.py 64


# GNNAdvisor, TCGNN, GeSpmm
cd eva/kernel/spmm
python spmm_baselines.py 64


