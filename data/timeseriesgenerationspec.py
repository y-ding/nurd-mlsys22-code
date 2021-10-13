import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from multiprocessing import Process, Pool
import sys
import os
from os import path
import csv

taskeventsdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/filtered_task_usage/'
outdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/timeseriesspec/'

def time_series(file):
    print("Working on File: " + str(file), flush = True)
    for chunk in pd.read_csv(taskeventsdirectory+'part-'+str(file).zfill(5)+'-of-00500.csv', chunksize=1000, index_col = False):
        chunk.fillna(0,inplace=True)
        for index, row in chunk.iterrows(): #['starttime', 'endtime', 'job ID', 'task index', 'machine ID', 'mean CPU usage','canonical memory usage','assigned memory usage','unmapped page cache memory usage','total page cache memory usage','maximum memory usage','mean disk I/O time','mean local disk space used','maximum CPU usage','maximum disk I/O time','cycles per instruction (CPI)','memory accesses per instruction (MAI)','sample portion','aggregation types','sampled CPU usage']
            #print("Aggregrating for "+ str(row['job ID']) +" " +str ([row['task index']]), flush = True)
            if int(row['job ID']) == 6283499093 or int(row['job ID']) == 6266469130 or int(row['job ID']) == 6274140245 or int(row['job ID']) == 6343946350 or int(row['job ID']) == 6308689702 or int(row['job ID']) == 6343048076:
                if not os.path.isdir(outdirectory+str(int(row['job ID']))):
                    os.makedirs(outdirectory+str(int(row['job ID'])))
                f = open(outdirectory+str(int(row['job ID']))+'/'+str(int(row['task index'])),"a+")
                f.write("%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n"%(row['starttime'],row['endtime'],row['job ID'], row['task index'], row['machine ID'], row['mean CPU usage'],row['canonical memory usage'],row['assigned memory usage'],row['unmapped page cache memory usage'],row['total page cache memory usage'],row['maximum memory usage'],row['mean disk I/O time'],row['mean local disk space used'],row['maximum CPU usage'],row['maximum disk I/O time'], row['cycles per instruction (CPI)'], row['memory accesses per instruction (MAI)'],row['sample portion'],row['aggregation types'],row['sampled CPU usage']))
                f.close()

def main():
    threadlist = []
    for i in range (0,500):
        time_series(i)

if __name__== "__main__":
    main()