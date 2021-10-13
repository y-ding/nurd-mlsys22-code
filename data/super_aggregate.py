import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from multiprocessing import Process, Pool
import sys
import os.path
import os
from os import path
import csv

basedirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/'
outdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/aggregateddata/'
outdirectoryfinal = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/aggregateddatafinal/'
taskstartoutfile = 'taskstarttimetasks.pickle'
tasksendoutfile = 'taskendttimetasks.pickle'

def main():
    aggregate_data= defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for filefolder in range (0,500):
        print("Working on File folder: " + str(filefolder), flush = True)
        for filename in os.listdir(outdirectory+str(filefolder)):
            for chunk in pd.read_csv(outdirectory+str(filefolder)+'/'+filename, chunksize=1000,  names=['job ID', 'task index', 'latency', 'mean CPU usage','canonical memory usage','assigned memory usage','unmapped page cache memory usage','total page cache memory usage','maximum memory usage','mean disk I/O time','mean local disk space used','maximum CPU usage','maximum disk I/O time','cycles per instruction (CPI)','memory accesses per instruction (MAI)','sampled CPU usage']):
                chunk.fillna(0,inplace=True)
                for index, row in chunk.iterrows(): #['starttime', 'endtime', 'job ID', 'task index', 'machine ID', 'mean CPU usage','canonical memory usage','assigned memory usage','unmapped page cache memory usage','total page cache memory usage','maximum memory usage','mean disk I/O time','mean local disk space used','maximum CPU usage','maximum disk I/O time','cycles per instruction (CPI)','memory accesses per instruction (MAI)','sample portion','aggregation types','sampled CPU usage']
            #print("Aggregrating for "+ str(row['job ID']) +" " +str ([row['task index']]), flush = True)
                    latency = (row['latency'])
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
    with open(basedirectory+taskstartoutfile, 'rb') as handle:
        tasksstartdictionary = pickle.load(handle)
    with open(basedirectory+tasksendoutfile, 'rb') as handle:
        tasksenddictionary = pickle.load(handle)
    for job in aggregate_data:
        with open(outdirectoryfinal+'job'+str(job)+'.csv', 'w') as f:
            for task in aggregate_data[job]:
                #print("Writing for "+ str(job) +" " +str(task), flush = True)
                if task in tasksenddictionary[job] and task in tasksstartdictionary[job]:
                    f.write("%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n"%(job,task,tasksenddictionary[job][task]-tasksstartdictionary[job][task],aggregate_data[job][task]['latency'],aggregate_data[job][task]['mean CPU usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['canonical memory usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['assigned memory usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['unmapped page cache memory usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['total page cache memory usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['maximum memory usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['mean disk I/O time']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['mean local disk space used']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['maximum CPU usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['maximum disk I/O time']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['cycles per instruction (CPI)']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['memory accesses per instruction (MAI)']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['sampled CPU usage']/aggregate_data[job][task]['latency']))
                else:
                     f.write("%d,%d,0,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n"%(job,task,aggregate_data[job][task]['latency'],aggregate_data[job][task]['mean CPU usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['canonical memory usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['assigned memory usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['unmapped page cache memory usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['total page cache memory usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['maximum memory usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['mean disk I/O time']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['mean local disk space used']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['maximum CPU usage']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['maximum disk I/O time']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['cycles per instruction (CPI)']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['memory accesses per instruction (MAI)']/aggregate_data[job][task]['latency'],aggregate_data[job][task]['sampled CPU usage']/aggregate_data[job][task]['latency']))
            f.close()


if __name__== "__main__":
    main()