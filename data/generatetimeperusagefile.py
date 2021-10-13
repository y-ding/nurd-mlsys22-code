import pandas as pd
import numpy as np
import pickle
import collections

taskeventsdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/filtered_task_usage/'

outdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/'
filestartoutfile = 'taskstarttime.pickle'
filesendoutfile = 'taskendttime.pickle'

def filestartandend():
    filestartdictionary=collections.defaultdict(dict)
    fileenddictionary=collections.defaultdict(dict)
    for ind in range(0,500):
        print ("Working on:"+str(ind), flush = True)
        df = pd.read_csv(taskeventsdirectory+'part-'+str(ind).zfill(5)+'-of-00500.csv')
        first = df.head(1)
        last = df.tail(1)
        filestartdictionary[ind]=first["starttime"]
        fileenddictionary[ind]=last["endtime"]
    with open(outdirectory+filestartoutfile, 'wb') as handle:
        pickle.dump(filestartdictionary, handle)
    with open(outdirectory+filesendoutfile, 'wb') as handle:
        pickle.dump(fileenddictionary, handle)

                

def main():
    filestartandend()

if __name__== "__main__":
    main()