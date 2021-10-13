import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from multiprocessing import Process, Pool
import sys
import os
from os import path
import csv

taskeventsdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/filtered_task_events/'
outdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/timeseriesschedule/'

def time_series(file):
    print("Working on File: " + str(file), flush = True)
    for chunk in pd.read_csv(taskeventsdirectory+'part-'+str(file).zfill(5)+'-of-00500.csv', chunksize=1000, index_col = False):
        chunk.fillna(0,inplace=True)
        for index, row in chunk.iterrows(): # names=['timestamp', 'missinginfo', 'job ID', 'task index', 'machine ID', 'event type','user name','scheduling class','priority','resource request for CPU cores','resource request for RAM','resource request for scratch disk space','different-machine constraint']
            if not os.path.isdir(outdirectory+str(int(row['job ID']))):
                os.makedirs(outdirectory+str(int(row['job ID'])))
            f = open(outdirectory+str(int(row['job ID']))+'/'+str(int(row['task index'])),"a+")
            f.write("%d,%d,%d,%d,%d,%d,%s,%f,%f,%f,%f,%f,%f\n"%(row['timestamp'],row['missinginfo'],row['job ID'], row['task index'], row['machine ID'], row['event type'],row['user name'],row['scheduling class'],row['priority'],row['resource request for CPU cores'],row['resource request for RAM'],row['resource request for scratch disk space'],row['different-machine constraint']))
            f.close()

def main():
    threadlist = []
    for i in range (0,500):
        time_series(i)

if __name__== "__main__":
    main()