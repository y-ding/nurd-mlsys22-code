import pandas as pd
import numpy as np
import pickle

jobeventsdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/job_events/'
taskeventsdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/task_events/'

outdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/'
jobSCoutfile = 'jobSC.pickle'
jobparallelismoutfile = 'jobparallelism.pickle'

def readinjobsSC():
    jobsdictionary={}
    for x in range(0,500):
        df = pd.read_csv(jobeventsdirectory+ 'part-'+str(x).zfill(5)+'-of-00500.csv', names=['timestamp', 'missinginfo', 'job ID', 'event type', 'user name', 'scheduling class','job name','logical job name'])
        for index, row in df.iterrows():
            if row['job ID'] not in jobsdictionary:
                jobsdictionary[row['job ID']] =row['scheduling class']
            else:
                if row['scheduling class'] > jobsdictionary[row['job ID']]:
                    jobsdictionary[row['job ID']] =row['scheduling class']
    return jobsdictionary

def readinjobsparallelism():
    jobsdictionary={}
    for x in range(0,500):
        df = pd.read_csv(taskeventsdirectory+'part-'+str(x).zfill(5)+'-of-00500.csv', names=['timestamp', 'missinginfo', 'job ID', 'task index', 'machine ID', 'event type','user name','scheduling class','priority','resource request for CPU cores','resource request for RAM','resource request for local disk space','different-machine constraint'])
        for index, row in df.iterrows():
            if row['job ID'] not in jobsdictionary:
                jobsdictionary[row['job ID']] =row['task index']+1
            else:
                if row['task index']+1 > jobsdictionary[row['job ID']]:
                    jobsdictionary[row['job ID']] =row['task index']+1
    return jobsdictionary

def main():
    jobsdictionary = readinjobsSC()
    with open(outdirectory+jobSCoutfile, 'wb') as handle:
        pickle.dump(jobsdictionary, handle)
    jobsdictionaryP = readinjobsparallelism()
    with open(outdirectory+jobparallelismoutfile, 'wb') as handle:
        pickle.dump(jobsdictionaryP, handle)
  
if __name__== "__main__":
    main()