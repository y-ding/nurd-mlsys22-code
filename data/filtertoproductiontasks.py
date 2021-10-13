import pandas as pd
import numpy as np
import pickle
from multiprocessing import Process, Pool
import sys
import os.path
from os import path

taskeventsdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/task_events/'


outdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/'
jobSCoutfile = 'jobSC.pickle'
jobparallelismoutfile = 'jobparallelism.pickle'
filteredoutdirectory = 'filtered_task_events/'

def filtertaskevents(schedulingclass, parallelism,low,high):
    for ind in range(low,high):
        df = pd.DataFrame()
        if not path.exists(outdirectory+filteredoutdirectory+'part-'+str(ind).zfill(5)+'-of-00500.csv'):
            print ("Working on:"+str(ind))
            for chunk in pd.read_csv(taskeventsdirectory+'part-'+str(ind).zfill(5)+'-of-00500.csv', names=['timestamp', 'missinginfo', 'job ID', 'task index', 'machine ID', 'event type','user name','scheduling class','priority','resource request for CPU cores','resource request for RAM','resource request for scratch disk space','different-machine constraint'],chunksize=1000):
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
    for t in range(0,25):
        threadlist.append(Process(target=filtertaskevents, args=(schedulingclass,parallelism,lower_bound+t*4,lower_bound+t*4+4)))       
    for thread in threadlist:
        thread.start()
    for thread in threadlist:
        thread.join()

  
if __name__== "__main__":
    main()