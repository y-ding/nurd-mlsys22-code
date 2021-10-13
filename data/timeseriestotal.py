import pandas as pd

import numpy as np
import pickle
from collections import defaultdict
from multiprocessing import Process, Pool
import sys
import os
from os import path
import csv

taskusagedirectory = './timeseries10/'
taskeventsdirectory = './timeseriesschedule/'
outdirectory = './timeseries10total/'

def time_series(file):
    try:
        df1 = pd.read_csv(taskusagedirectory+file, index_col = False, names = ['starttime', 'endtime', 'job ID', 'task index', 'machine ID', 'mean CPU usage','canonical memory usage','assigned memory usage','unmapped page cache memory usage','total page cache memory usage','maximum memory usage','mean disk I/O time','mean local disk space used','maximum CPU usage','maximum disk I/O time','cycles per instruction (CPI)','memory accesses per instruction (MAI)','sample portion','aggregation types','sampled CPU usage'])
        df1.fillna(0,inplace=True)
        df2 = pd.read_csv(taskeventsdirectory+file, index_col = False, names = ['timestamp', 'missinginfo', 'job ID', 'task index', 'machine ID', 'event type','user name','scheduling class','priority','resource request for CPU cores','resource request for RAM','resource request for scratch disk space','different-machine constraint'])
        df2.fillna(0,inplace=True)
        iter2 = df2.iterrows()
        nextrow = next(iter2,(0,pd.Series([])))[1]
        for index,row in df1.iterrows():
            starttime = int(row['starttime'])
            endtime = int(row['endtime'])
            evict_count = 0
            fail_count = 0
            while True:
                #print("Row: " + str(nextrow), flush=True)
                if nextrow.empty:
                    break
                if int(nextrow['timestamp']) >= endtime:
                    break
                if int(nextrow['event type']) == 2:
                    evict_count = evict_count + 1
                if int(nextrow['event type']) == 3:
                    fail_count = fail_count + 1
                nextrow = next(iter2,(0, pd.Series([])))[1]
                if nextrow.empty:
                    break
            if not os.path.isdir(outdirectory+str(int(row['job ID']))):
                os.makedirs(outdirectory+str(int(row['job ID'])))
            f = open(outdirectory+str(int(row['job ID']))+'/'+str(int(row['task index'])),"a+")
            f.write("%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d\n"%(int(row['starttime']),int(row['endtime']),int(row['job ID']), int(row['task index']), int(row['machine ID']), float(row['mean CPU usage']),float(row['canonical memory usage']),float(row['assigned memory usage']),float(row['unmapped page cache memory usage']),float(row['total page cache memory usage']),float(row['maximum memory usage']),float(row['mean disk I/O time']),float(row['mean local disk space used']),float(row['maximum CPU usage']),float(row['maximum disk I/O time']), float(row['cycles per instruction (CPI)']), float(row['memory accesses per instruction (MAI)']),float(row['sample portion']),float(row['aggregation types']),float(row['sampled CPU usage']),evict_count,fail_count))
            f.close()
    except:
        return
        
    


def main():
    for filename in os.listdir(taskusagedirectory):
        for file in os.listdir(taskusagedirectory + filename):
            time_series(filename + '/'+file)

if __name__== "__main__":
    main()
