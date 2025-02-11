#!/usr/bin/env python3
import sys
import os
import numpy as np
from scipy.io import mmread, mmwrite
import scipy.sparse

def get_block_indices(total, num_blocks):
    """
    根据总数和块数计算每块的起始和结束索引，
    如果不能整除，则前面块多分配1个单位。
    """
    block_size = total // num_blocks
    remainder = total % num_blocks
    indices = []
    start = 0
    for i in range(num_blocks):
        extra = 1 if i < remainder else 0
        end = start + block_size + extra
        indices.append((start, end))
        start = end
    return indices

def split_sparse_matrix(mat, num_row_blocks, num_col_blocks, base_filename):
    """
    将稀疏矩阵 mat 按照行和列分别分成 num_row_blocks 和 num_col_blocks 块，
    并将每个块写成 .mtx 文件。
    """
    # 为了高效切片，确保使用 CSR 格式
    if not scipy.sparse.isspmatrix_csr(mat):
        mat = mat.tocsr()
    row_ranges = get_block_indices(mat.shape[0], num_row_blocks)
    col_ranges = get_block_indices(mat.shape[1], num_col_blocks)
    for i, (r_start, r_end) in enumerate(row_ranges):
        for j, (c_start, c_end) in enumerate(col_ranges):
            submat = mat[r_start:r_end, c_start:c_end]
            out_filename = f"{base_filename}_block_{i}_{j}.mtx"
            mmwrite(out_filename, submat)
            print(f"Wrote sparse block ({i}, {j}) to {out_filename}")

def split_dense_matrix(mat, num_row_blocks, num_col_blocks, base_filename):
    """
    将密集矩阵 mat 按照行和列分别分成 num_row_blocks 和 num_col_blocks 块，
    并将每个块写成 .mtx 文件。
    """
    row_ranges = get_block_indices(mat.shape[0], num_row_blocks)
    col_ranges = get_block_indices(mat.shape[1], num_col_blocks)
    for i, (r_start, r_end) in enumerate(row_ranges):
        for j, (c_start, c_end) in enumerate(col_ranges):
            submat = mat[r_start:r_end, c_start:c_end]
            out_filename = f"{base_filename}_block_{i}_{j}.mtx"
            mmwrite(out_filename, submat)
            print(f"Wrote dense block ({i}, {j}) to {out_filename}")

def main():
    if len(sys.argv) != 5:
        print("Usage: python devide.py path/to/.mtx Z X Y")
        sys.exit(1)
    mtx_path = sys.argv[1]
    try:
        Z = int(sys.argv[2])
        X = int(sys.argv[3])
        Y = int(sys.argv[4])
    except ValueError:
        print("Z, X, and Y must be integers.")
        sys.exit(1)
    
    # 读取 Matrix Market 文件
    print(f"Reading matrix from {mtx_path}...")
    mat = mmread(mtx_path)
    print(f"Matrix shape: {mat.shape}")
    
    # 取掉文件扩展名作为基础文件名
    base_filename = os.path.splitext(mtx_path)[0]
    
    # 如果是稀疏矩阵，按照 A 的分块方式 (Z 行块, X 列块)
    if scipy.sparse.isspmatrix(mat):
        print(f"Detected sparse matrix. Splitting into {Z} row blocks and {X} column blocks (for matrix A).")
        split_sparse_matrix(mat, Z, X, base_filename)
    else:
        # 如果是密集矩阵，按照 H 的分块方式 (X 行块, Y 列块)
        print(f"Detected dense matrix. Splitting into {X} row blocks and {Y} column blocks (for matrix H).")
        split_dense_matrix(mat, X, Y, base_filename)

if __name__ == "__main__":
    main()

