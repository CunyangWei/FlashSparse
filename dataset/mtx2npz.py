import argparse
import numpy as np
from scipy.io import mmread
from scipy.sparse import coo_matrix

parser = argparse.ArgumentParser(description='Convert .mtx to TCGNN-compatible .npz')
parser.add_argument("input", help="Path to input .mtx file")
parser.add_argument("output", help="Path to output .npz file")
args = parser.parse_args()

coo = mmread(args.input)
row = coo.row
col = coo.col
num_edges = coo.nnz
num_nodes_src = coo.shape[0]
num_nodes_dst = coo.shape[1]

data_dict = {
    'num_nodes_src': np.array(num_nodes_src, dtype=np.int32),
    'num_nodes_dst': np.array(num_nodes_dst, dtype=np.int32),
    'num_edges':     np.array(num_edges,     dtype=np.int32),
    'src_li':        row.astype(np.int32),
    'dst_li':        col.astype(np.int32)
}

np.savez(args.output, **data_dict)

print(f"Converted {args.input} to {args.output} (TCGNN-compatible format).")

