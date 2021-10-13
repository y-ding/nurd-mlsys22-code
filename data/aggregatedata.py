import pandas as pd
import numpy as np
import pickle
import collections
from multiprocessing import Process, Pool
import sys
import os.path
from os import path

taskeventsdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/filtered_task_usage/'


outdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/'
taskstartoutfile = 'taskstarttimetasks.pickle'
tasksendoutfile = 'taskendttimetasks.pickle'
filestartoutfile = 'taskstarttime.pickle'
filesendoutfile = 'taskendttime.pickle'
finalcsv = 'aggregateddata.csv'
outdirectoryfinal = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/aggregateddata/'

def complete_single_aggregration(job, tasksstartdictionary, tasksenddictionary, filesstartdictionary, filesenddictionary):
    for task in tasksstartdictionary:
        start = tasksstartdictionary[task]
        end = tasksenddictionary[task]
        print("working on job: "+ str(job)+ " task: " +str(task),flush=True)
        startfile = -1
        endfile = -1
        for ind in range(0,500):
            print("Working on IND "+str(ind),flush=True)
            if filesstartdictionary[ind] <= start and filesenddictionary[ind] >= start:
                startfile = ind
            if filesstartdictionary[ind] <= end and filesenddictionary[ind] >= end:
                endfile = ind
            if startfile != -1 and endfile != -1:
                break
        print("found files for job:"+ str(job)+ " task: " +str(task),flush=True)
        TIMEINTERVAL = 0
        TOTALTIME = 0
        MCU = 0.0
        CMU = 0.0
        AMU = 0.0
        UPCMU = 0.0
        TPCMU = 0.0
        MMU = 0.0
        MDIT = 0.0
        MDLSU = 0.0
        MAXCPU = 0.0
        MAXDIT = 0.0
        CPI = 0.0
        MAPI = 0.0
        SPU = 0.0
        for ind in range(startfile,endfile):
            for chunk in pd.read_csv(taskeventsdirectory+'part-'+str(ind).zfill(5)+'-of-00500.csv', chunksize=1000):
                for index, row in chunk.iterrows(): #['starttime', 'endtime', 'job ID', 'task index', 'machine ID', 'mean CPU usage','canonical memory usage','assigned memory usage','unmapped page cache memory usage','total page cache memory usage','maximum memory usage','mean disk I/O time','mean local disk space used','maximum CPU usage','maximum disk I/O time','cycles per instruction (CPI)','memory accesses per instruction (MAI)','sample portion','aggregation types','sampled CPU usage']
                    if job == row['job id'] and task == row['task index']:
                        TIMEINTERVAL = row['startime']-['endtime']
                        TOTALTIME = TOTALTIME + TIMEINTERVAL
                        MCU = MCU + row['mean CPU usage']*TIMEINTERVAL
                        CMU = CMU + row['canonical memory usage']*TIMEINTERVAL
                        AMU = AMU + row['assigned memory usage']*TIMEINTERVAL
                        UPCMU = UPCMU + row['unmapped page cache memory usage']*TIMEINTERVAL
                        TPCMU = TPCMU + row['total page cache memory usage']*TIMEINTERVAL
                        MMU = MMU + row['maximum memory usage']*TIMEINTERVAL
                        MDIT = MDIT + row['mean disk I/O time']*TIMEINTERVAL
                        MDLSU = MDLSU + row['mean local disk space used']*TIMEINTERVAL
                        MAXCPU = MAXCPU + row['maximum CPU usage']*TIMEINTERVAL
                        MAXDIT = MAXDIT + row['maximum disk I/O time']*TIMEINTERVAL
                        CPI = CPI + row['cycles per instruction (CPI)']*TIMEINTERVAL
                        MAPI = MAPI + row['memory accesses per instruction (MAI)']*TIMEINTERVAL
                        SPU = SPU + row['sampled CPU usage']*TIMEINTERVAL
        data = {'JOBID':[job],'TASKINDEX':[task],'LATENCY':[end-start],'MCU':[MCU/TOTALTIME],'CMU':[CMU/TOTALTIME],'AMU':[AMU/TOTALTIME],'UPCMU':[UPCMU/TOTALTIME],'TPCMU':[TPCMU/TOTALTIME],'MMU':[MMU/TOTALTIME],'MDIT':[MDIT/TOTALTIME],'MDLSU':[MDLSU/TOTALTIME],'MAXCPU':[MAXCPU/TOTALTIME],'MAXDIT':[MAXDIT/TOTALTIME],'CPI':[CPI/TOTALTIME],'MAPI':[MAPI/TOTALTIME],'SPU':[MCU/TOTALTIME]} 
        dataframe = pd.DataFrame(data) 
        print("about to write "+ str(job)+ " task: " +str(task),flush=True)
        with open(outdirectory+str(job)+finalcsv, 'a+') as f:
            dataframe.to_csv(f, header=False)
        print("finished working on job: "+ str(job)+ " task: " +str(task),flush=True)
    print("COMPLETELY finished working on job: "+ str(job))

def main():
    with open(outdirectory+taskstartoutfile, 'rb') as handle:
        tasksstartdictionary = pickle.load(handle)
    with open(outdirectory+tasksendoutfile, 'rb') as handle:
        tasksenddictionary = pickle.load(handle)
    with open(outdirectory+filestartoutfile, 'rb') as handle:
        filesstartdictionary = pickle.load(handle)
    with open(outdirectory+filesendoutfile, 'rb') as handle:
        filesenddictionary = pickle.load(handle)
    pool = Pool(processes=10) 
    i = 0
    for job in tasksstartdictionary:
        i = i + 1
        pool.apply_async(complete_single_aggregration, (job, tasksstartdictionary[job], tasksenddictionary[job], filesstartdictionary, filesenddictionary))
        if i == 10:        
            pool.close()
            pool.join()
            pool = Pool(processes=10) 
            i = 0

if __name__== "__main__":
    main()

