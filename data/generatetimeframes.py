import pandas as pd
import numpy as np
import pickle
import collections

taskeventsdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/filtered_task_events/'

outdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/'
taskstartoutfile = 'taskstarttimetasks.pickle'
tasksendoutfile = 'taskendttimetasks.pickle'

def taskstart():
    taskstartdictionary=collections.defaultdict(dict)
    for ind in range(0,500):
        df = pd.DataFrame()
        print ("Working on:"+str(ind), flush = True)
        for chunk in pd.read_csv(taskeventsdirectory+'part-'+str(ind).zfill(5)+'-of-00500.csv',chunksize=1000):
            for index, row in chunk.iterrows():
                if row['event type'] == 0:
                    taskstartdictionary[row['job ID']][row['task index']]=row['timestamp']
                    #print ('Job ID, Task Index, Start: '+str(row['job ID'])+","+str(row['task index'])+","+str(row['timestamp']))
    return taskstartdictionary

def taskend():
    tasksenddictionary=collections.defaultdict(dict)
    for ind in range(0,500):
        df = pd.DataFrame()
        print ("Working on:"+str(ind), flush = True)
        for chunk in pd.read_csv(taskeventsdirectory+'part-'+str(ind).zfill(5)+'-of-00500.csv',chunksize=1000):
            for index, row in chunk.iterrows():
                if row['event type'] == 4:
                    tasksenddictionary[row['job ID']][row['task index']]=row['timestamp']
                    #print ('Job ID, Task Index, End: '+str(row['job ID'])+","+str(row['task index'])+","+str(row['timestamp']))
    return tasksenddictionary

                

def main():
    taskstartdictionary = taskstart()
    with open(outdirectory+taskstartoutfile, 'wb') as handle:
        pickle.dump(taskstartdictionary, handle)
    tasksenddictionary = taskend()
    with open(outdirectory+tasksendoutfile, 'wb') as handle:
        pickle.dump(tasksenddictionary, handle)
  
if __name__== "__main__":
    main()