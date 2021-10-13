import pandas as pd
import numpy as np
import pickle
from multiprocessing import Process, Pool
import sys
import os.path
from os import path

taskeventsdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/task_usage/'


outdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/'
jobSCoutfile = 'jobSC.pickle'
jobparallelismoutfile = 'jobparallelism.pickle'
filteredoutdirectory = 'filtered_task_usage/'


def filtertaskusageevents(schedulingclass, parallelism,low,high):
    for ind in range(low,high):
        df = pd.DataFrame()
        if not path.exists(outdirectory+filteredoutdirectory+'part-'+str(ind).zfill(5)+'-of-00500.csv'):
            print ("Working on:"+str(ind))
            for chunk in pd.read_csv(taskeventsdirectory+'part-'+str(ind).zfill(5)+'-of-00500.csv', names=['starttime', 'endtime', 'job ID', 'task index', 'machine ID', 'mean CPU usage','canonical memory usage','assigned memory usage','unmapped page cache memory usage','total page cache memory usage','maximum memory usage','mean disk I/O time','mean local disk space used','maximum CPU usage','maximum disk I/O time','cycles per instruction (CPI)','memory accesses per instruction (MAI)','sample portion','aggregation types','sampled CPU usage'],chunksize=1000):
                for index, row in chunk.iterrows():
                    x = row['job ID'] 
                    if x in schedulingclass and x in parallelism:
                        if schedulingclass[x] == 0 or parallelism [x] < 100:
                            chunk.drop(index, inplace=True)
                    else:
                        chunk.drop(index, inplace=True)
                df = pd.concat([df,chunk])
            df.to_csv(outdirectory+filteredoutdirectory+'part-'+str(ind).zfill(5)+'-of-00500.csv')
                
def main():
    lower_bound = int(sys.argv[1])
    with open(outdirectory+jobSCoutfile, 'rb') as handle:
        schedulingclass = pickle.load(handle)
    with open(outdirectory+jobparallelismoutfile, 'rb') as handle:
        parallelism = pickle.load(handle)
    threadlist = []
    for t in range(0,10):
        threadlist.append(Process(target=filtertaskusageevents, args=(schedulingclass,parallelism,lower_bound+t*10,lower_bound+t*10+10)))       
    for thread in threadlist:
        thread.start()
    
    for thread in threadlist:
        thread.join()
      
if __name__== "__main__":
    main()