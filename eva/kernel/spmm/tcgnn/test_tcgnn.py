import sys
sys.path.append('./eva100/kernel/gcn')
from tcgnn.mdataset import *
import time
import TCGNN_kernel
import torch
from scipy.io import mmread
import numpy as np


def kernel(inputInfo, epoches):
    X_prime, spmm_ms_avg = TCGNN_kernel.forward(inputInfo.x, inputInfo.values, inputInfo.row_pointers, inputInfo.column_index, inputInfo.blockPartition, inputInfo.edgeToColumn, inputInfo.edgeToRow, epoches)

    return X_prime, round(spmm_ms_avg.item(), 4)

def test(data, epoches, dimN, data_path):
    # 记录程序开始时间
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data, data_path)
    inputInfo.init_embedding(dimN)
    inputInfo = inputInfo.to(device)

    X_prime, execution_time = kernel(inputInfo, epoches)
    '''
    print(data)
    print(data_path)
    
    mtx_path = data_path.replace(".npz", ".mtx")
    print("Reading sparse matrix from", mtx_path)
    A = mmread(mtx_path)
    B = inputInfo.x.cpu().numpy()
    computed = A.dot(B)
    X_prime_np = X_prime.cpu().numpy() if X_prime.is_cuda else X_prime.numpy()

    if np.allclose(computed, X_prime_np, rtol=1e-4, atol=1e-6):
        print("Sparse matrix multiplication result matches X_prime.")
    else:
        diff_norm = np.linalg.norm(computed - X_prime_np)
        print(f"Results differ, difference norm: {diff_norm}")
    '''
    print(str(dimN) + '-' + data + ' tcgnn-' + str(execution_time))
    return execution_time

