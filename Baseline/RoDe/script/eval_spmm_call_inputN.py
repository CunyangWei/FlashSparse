import torch
import pandas as pd
import csv
import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import sys

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
data_dir = os.getenv('SPMM_DATADIR', '/home/cunyang/workspace/dataset/')
inputN = sys.argv[1]
GPU_name = '0'
if len(sys.argv) > 2:
    GPU_name = sys.argv[2]
else:
    GPU_name = '0'
# df = pd.read_csv(project_dir + '/dataset/data_filter.csv')
# df = pd.read_csv(project_dir + '/result/ref/baseline_h100_spmm_128.csv')
df = pd.read_csv(project_dir + '/dataset/data.csv')
#result path
file_name = project_dir + '/result/Baseline/spmm/' + GPU_name + '_rode_sputnik_cusparse_spmm_f32_n' + inputN + '.csv'
head = ['dataSet','rows_','columns_','nonzeros_','sputnik','Sputnik_gflops','cusparse','cuSPARSE_gflops','rode','ours_gflops']

with open(file_name, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(head)
count = 0

start_time = time.time()
for index, row in df.iterrows():
    count+=1
    print(index)
    print(row)
    
    data = [row['dataSet']]
    with open(file_name, 'a', newline='') as csvfile:
        csvfile.write(','.join(map(str, data)))
        
    # shell_command = project_dir + "/Baseline/RoDe/build/eval/eval_spmm_f32_n128 " + project_dir + "/Baseline/RoDe/dataset/" + row['dataSet'] + '/' + row['dataSet'] + ".mtx >> " + file_name
    shell_command = project_dir + "/Baseline/RoDe/build/eval/eval_spmm_f32_inputN " + inputN + " " + data_dir + row['dataSet'] + '/' + row['dataSet'] + ".mtx >> " + file_name
    print(row['dataSet'])
    subprocess.run(shell_command, shell=True)

end_time = time.time()
execution_time = end_time - start_time

print("Execution time: ", execution_time)

dimN = 128
# Record execution time.
# with open("execution_time_base.txt", "a") as file:
#     file.write("spmm-" + str(dimN) + "-" + str(round(execution_time/60,2)) + " minutes\n")
