import os
import sys
from fs_tf32.mdataset2 import *
import FS_SpMM
import torch
from scipy.io import mmread
import numpy as np

# 8x1
def fs_tf32_8_1(data, epoches, dimN, partsize_t, data_path,  window, wide):

    inputInfo = dataSet_tf32(data, dimN, partsize_t, data_path, window, wide)
    
    X_prime, spmm_ms_avg  = FS_SpMM.forward_tf32(   
        inputInfo.row_pointers, 
        inputInfo.column_index, 
        inputInfo.degrees.float(), 
        inputInfo.x, 
        inputInfo.num_nodes, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, epoches)
    '''
    mtx_path = data_path + "/" + data + ".mtx"
    print("Reading sparse matrix from", mtx_path)
    A = mmread(mtx_path)

    B = inputInfo.x.cpu().numpy()

    computed = A.dot(B)

    X_prime_np = X_prime.cpu().numpy() if X_prime.is_cuda else X_prime.numpy()

    if np.allclose(computed, X_prime_np, rtol=1e-4, atol=1e-2):
        print("Sparse matrix multiplication result matches X_prime.")
    else:
        diff_norm = np.linalg.norm(computed - X_prime_np)
        print(f"Results differ, difference norm: {diff_norm}")
    print(X_prime_np)
    print(computed)
    '''
    spmm_ms_avg = round((spmm_ms_avg.item()),2)
    print(str(dimN) + '-' + data + 'tcu_8_1' + '-' +str(spmm_ms_avg))
    return X_prime, spmm_ms_avg

def fs_tf32_8_1_map(data, epoches, dimN, partsize_t, data_path,  window, wide):

    inputInfo = dataSet_tf32(data, dimN, partsize_t, data_path, window, wide)
    
    X_prime, spmm_ms_avg  = FS_SpMM.forward_tf32_map(   
        inputInfo.row_pointers, 
        inputInfo.column_index, 
        inputInfo.degrees.float(), 
        inputInfo.x, 
        inputInfo.num_nodes, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, epoches)
    '''
    mtx_path = data_path + "/" + data + ".mtx"
    print("Reading sparse matrix from", mtx_path)
    A = mmread(mtx_path)

    B = inputInfo.x.cpu().numpy()
    
    computed = A.dot(B)

    X_prime_np = X_prime.cpu().numpy() if X_prime.is_cuda else X_prime.numpy()

    if np.allclose(computed, X_prime_np, rtol=1e-4, atol=1e-2):
        print("\033[92mSparse matrix multiplication result matches X_prime.\033[0m")
    else:
        diff_norm = np.linalg.norm(computed - X_prime_np)
        print(f"\033[91mResults differ, difference norm: {diff_norm}\033[0m")
    print(X_prime_np)
    print(computed)
    '''
    spmm_ms_avg = round((spmm_ms_avg.item()),2)
    print(str(dimN) + '-' + data + 'tcu_8_1_map' + '-' +str(spmm_ms_avg))
    return spmm_ms_avg

# 8x1_balance
def fs_tf32_8_1_balance(data, epoches, dimN, partsize_t, data_path,  window, wide):

    inputInfo = dataSet_tf32_balance(data, dimN, partsize_t, data_path, window, wide)
    
    X_prime, spmm_ms_avg  = FS_SpMM.forward_tf32_balance(   
        inputInfo.row_pointers, 
        inputInfo.column_index, 
        inputInfo.degrees.float(), 
        inputInfo.t_window_rowTensor,
        inputInfo.t_atomicTensor,
        inputInfo.x, 
        inputInfo.num_nodes, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, epoches)

    spmm_ms_avg = round((spmm_ms_avg.item()),2)
    print(str(dimN) + '-' + data + 'tcu_8_1_balance' + '-' +str(spmm_ms_avg))
    return spmm_ms_avg


def fs_tf32_16_1(data, epoches, dimN, partsize_t, data_path,  window, wide):

    inputInfo = dataSet_tf32(data, dimN, partsize_t, data_path, window, wide)


    X_prime, spmm_ms_avg  = FS_SpMM.forward_tf32_16(   
        inputInfo.row_pointers, 
        inputInfo.column_index, 
        inputInfo.degrees.float(), 
        inputInfo.x, 
        inputInfo.num_nodes, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, epoches)
    spmm_ms_avg = round((spmm_ms_avg.item()),2)
    print(str(dimN) + '-' + data + 'tcu_16_1_test' + '-' +str(spmm_ms_avg))
    return spmm_ms_avg
    # res.append(spmm_ms_avg)
    # with open(file, 'a', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow(res)