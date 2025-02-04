# Summarize Rode, Gespmm, Sputnik, Advisor, DTC, TCGNN, cuSPARSE
import scipy.sparse as sp
from scipy.io import mmread
import pandas as pd
import os
def mean(input_list):
    return round((sum(input_list) / len(input_list)),1)
#dimN_list = [128, 256]
dimN_list = [64]

data_dir = os.getenv('SPMM_DATADIR', '/home/cunyang/workspace/dataset/')

project_dir = '/home/cunyang/scratch.bhatele-lab/FlashSparse/'
base_result_dir = project_dir + 'result/Baseline/spmm'
for dimN in dimN_list:
    df1 = pd.read_csv(base_result_dir + '/rode_sputnik_cusparse_spmm_f32_n' + str(dimN) + '.csv')
    df2 = pd.read_csv(base_result_dir + '/advisor_tcgnn_gespmm_f32_n' + str(dimN) + '.csv')
    df_res = pd.merge(df1, df2, on='dataSet', how='inner')
    df_res1 = df_res[['dataSet', 'num_nodes', 'num_edges', 'sputnik',
                      'cusparse', 'rode', 'gespmm', 'advisor', 'tcgnn']]
    
    df_metadata = pd.read_csv(project_dir + 'dataset/data.csv')
    df_res1 = pd.merge(df_res1, df_metadata[['dataSet','NNZ']], on='dataSet', how='left')

    libs = ['sputnik','cusparse','rode','gespmm','advisor','tcgnn']
    for lib in libs:
        df_res1['GFLOPS_' + lib] = df_res1.apply(
            lambda row: (2 * row['NNZ'] * dimN) / (row[lib] * 1e6)
                        if row[lib] and row['NNZ'] is not None and row[lib] > 0 else None,
            axis=1)
    dtc = pd.read_csv(base_result_dir + '/my_dtc_spmm_f32_n' + str(dimN) + '.csv')[['dataSet', 'dtc']]
    df_res2 = pd.merge(df_res1, dtc, on='dataSet', how='inner')
    df_res2['GFLOPS_dtc'] = df_res2.apply(
        lambda row: (2 * row['NNZ'] * dimN) / (row['dtc'] * 1e6)
                    if row['dtc'] and row['NNZ'] is not None and row['dtc'] > 0 else None,
        axis=1)
    df_res2 = df_res2[['dataSet', 'num_nodes', 'num_edges', 'NNZ',
                        'sputnik','GFLOPS_sputnik', 'cusparse','GFLOPS_cusparse',
                        'rode','GFLOPS_rode', 'gespmm','GFLOPS_gespmm',
                        'advisor','GFLOPS_advisor', 'tcgnn','GFLOPS_tcgnn',
                        'dtc','GFLOPS_dtc']]
    df_res2.to_csv(project_dir + 'result' + '/all_baseline_' + str(dimN) + '.csv', index=False)

    df_flash_dim = pd.read_csv(project_dir + 'result/FlashSparse/spmm/flashsparse_tf32_' + str(dimN) + '.csv')
    print(df_flash_dim)
    libs_dim = ['16_1','8_1','8_1_balance','8_1_map']
    nnz_mapping = dict(zip(df_res2['dataSet'], df_res2['NNZ']))
    df_flash_dim['NNZ_flash_dim'] = df_flash_dim['dataSet'].map(nnz_mapping)

    for lib in libs_dim:
        df_flash_dim['GFLOPS_' + lib + '_flash_dim'] = df_flash_dim.apply(
            lambda row: (2 * row['NNZ_flash_dim'] * dimN) / (row[lib] * 1e6)
                        if row[lib] and row['NNZ_flash_dim'] is not None and row[lib] > 0 else None,
            axis=1)
    df_base = pd.read_csv(project_dir + 'result' + '/all_baseline_' + str(dimN) + '.csv')
    df_merged = pd.merge(df_base, df_flash_dim, on=['dataSet','num_nodes','num_edges'], how='outer')
    df_merged.to_csv(project_dir + 'result'  + '/all_baseline_' + str(dimN) + '.csv', index=False)
