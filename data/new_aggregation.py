import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from multiprocessing import Process, Pool
import sys
import os.path
from os import path
import csv

taskeventsdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/filtered_task_usage/'
outdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/aggregateddata/'

def aggregate_over_file(file):
    aggregate_data= defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    print("Working on File: " + str(file), flush = True)
    for chunk in pd.read_csv(taskeventsdirectory+'part-'+str(file).zfill(5)+'-of-00500.csv', chunksize=1000, index_col = False):
        chunk.fillna(0,inplace=True)
        for index, row in chunk.iterrows(): #['starttime', 'endtime', 'job ID', 'task index', 'machine ID', 'mean CPU usage','canonical memory usage','assigned memory usage','unmapped page cache memory usage','total page cache memory usage','maximum memory usage','mean disk I/O time','mean local disk space used','maximum CPU usage','maximum disk I/O time','cycles per instruction (CPI)','memory accesses per instruction (MAI)','sample portion','aggregation types','sampled CPU usage']
            #print("Aggregrating for "+ str(row['job ID']) +" " +str ([row['task index']]), flush = True)
            latency = (row['endtime']-row['starttime'])
            aggregate_data[row['job ID']][row['task index']]['latency'] = aggregate_data[row['job ID']][row['task index']]['latency'] + latency
            aggregate_data[row['job ID']][row['task index']]['mean CPU usage'] = aggregate_data[row['job ID']][row['task index']]['mean CPU usage'] + row['mean CPU usage']*latency
            aggregate_data[row['job ID']][row['task index']]['canonical memory usage'] = aggregate_data[row['job ID']][row['task index']]['canonical memory usage'] + row['canonical memory usage']*latency
            aggregate_data[row['job ID']][row['task index']]['assigned memory usage'] = aggregate_data[row['job ID']][row['task index']]['assigned memory usage'] + row['assigned memory usage']*latency
            aggregate_data[row['job ID']][row['task index']]['unmapped page cache memory usage'] = aggregate_data[row['job ID']][row['task index']]['unmapped page cache memory usage'] + row['unmapped page cache memory usage']*latency
            aggregate_data[row['job ID']][row['task index']]['total page cache memory usage'] = aggregate_data[row['job ID']][row['task index']]['total page cache memory usage'] + row['total page cache memory usage']*latency
            aggregate_data[row['job ID']][row['task index']]['assigned memory usage'] = aggregate_data[row['job ID']][row['task index']]['assigned memory usage'] + row['assigned memory usage']*latency
            aggregate_data[row['job ID']][row['task index']]['maximum memory usage'] = aggregate_data[row['job ID']][row['task index']]['maximum memory usage'] + row['maximum memory usage']*latency
            aggregate_data[row['job ID']][row['task index']]['mean disk I/O time'] = aggregate_data[row['job ID']][row['task index']]['mean disk I/O time'] + row['mean disk I/O time']*latency
            aggregate_data[row['job ID']][row['task index']]['mean local disk space used'] = aggregate_data[row['job ID']][row['task index']]['mean local disk space used'] + row['mean local disk space used']*latency
            aggregate_data[row['job ID']][row['task index']]['maximum CPU usage'] = aggregate_data[row['job ID']][row['task index']]['maximum CPU usage'] + row['maximum CPU usage']*latency
            aggregate_data[row['job ID']][row['task index']]['maximum disk I/O time'] = aggregate_data[row['job ID']][row['task index']]['maximum disk I/O time'] + row['maximum disk I/O time']*latency
            aggregate_data[row['job ID']][row['task index']]['cycles per instruction (CPI)'] = aggregate_data[row['job ID']][row['task index']]['cycles per instruction (CPI)'] + row['cycles per instruction (CPI)']*latency
            aggregate_data[row['job ID']][row['task index']]['memory accesses per instruction (MAI)'] = aggregate_data[row['job ID']][row['task index']]['memory accesses per instruction (MAI)'] + row['memory accesses per instruction (MAI)']*latency
            aggregate_data[row['job ID']][row['task index']]['sampled CPU usage'] = aggregate_data[row['job ID']][row['task index']]['maximum memory usage'] + row['sampled CPU usage']*latency
    for job in aggregate_data:
        with open(outdirectory+str(file)+'/'+'job'+str(job)+'.csv', 'w') as f:
            for task in aggregate_data[job]:
                #print("Writing for "+ str(job) +" " +str(task), flush = True)
                f.write("%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n"%(job,task,aggregate_data[job][task]['latency'],aggregate_data[job][task]['mean CPU usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['canonical memory usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['assigned memory usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['unmapped page cache memory usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['total page cache memory usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['maximum memory usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['mean disk I/O time']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['mean local disk space used']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['maximum CPU usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['maximum disk I/O time']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['cycles per instruction (CPI)']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['memory accesses per instruction (MAI)']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['sampled CPU usage']/aggregate_data[job][task]['latency']))
            f.close()

def main():
    file = int(sys.argv[1])
    threadlist = []
    for i in range (0,50):
        aggregate_over_file(file+i)

if __name__== "__main__":
    main()