import pandas as pd
import numpy as np
import pickle

jobeventsdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/job_events/'


outdirectory = '/scratch/avinashrao/causal-inference/clusterdata-2011-2/'
jobSCoutfile = 'jobSC.pickle'
jobparallelismoutfile = 'jobparallelism.pickle'
filteredoutdirectory = 'filtered_job_events/'

def filterjobevents():
    with open(outdirectory+jobSCoutfile, 'rb') as handle:
        schedulingclass = pickle.load(handle)
    with open(outdirectory+jobparallelismoutfile, 'rb') as handle:
        parallelism = pickle.load(handle)
    for ind in range(0,500):
        df = pd.read_csv(jobeventsdirectory+ 'part-'+str(ind).zfill(5)+'-of-00500.csv', names=['timestamp', 'missinginfo', 'job ID', 'event type', 'user name', 'scheduling class','job name','logical job name'])
        for index, row in df.iterrows():
             x = row['job ID'] 
             if x in schedulingclass and x in parallelism:
                if schedulingclass[x] == 0 or parallelism [x] < 20:
                    df.drop(index, inplace=True)
             else:
                df.drop(index, inplace=True)
        df.to_csv(outdirectory+filteredoutdirectory+'part-'+str(ind).zfill(5)+'-of-00500.csv')
                
def main():
    filterjobevents()
  
if __name__== "__main__":
    main()