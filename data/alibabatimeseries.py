import numpy as np
import pickle
from collections import defaultdict
from multiprocessing import Process, Pool
import sys
import os
from os import path
import csv
import pandas as pd

batchfile = './batch_instance.csv'
outdirectory = './alibabatimeseries2018filtered/'

def main():
    for chunk in pd.read_csv(batchfile, chunksize=1000, index_col = False, names = ['instance_name','task ID','job ID','task_type','status','starttime', 'endtime', 'machine ID','sequence number','total sequence number','average CPU','maximum CPU','average memory','maximum memory']):
        chunk.fillna(0,inplace=True) 
        for index, row in chunk.iterrows():
            if row['status']=='TERMINATED':
                f = open(outdirectory+str(row['job ID']+row['task ID']),"a+")
                f.write("%d,%d,%s,%s,%s,%s,%d,%d,%f,%f,%f,%f\n"%(row['starttime'],row['endtime'],row['job ID'],row['task ID'],row['machine ID'],row['status'],row['sequence number'],row['total sequence number'],row['maximum CPU'],row['average CPU'],row['maximum memory'],row['average memory']))
                f.close()         

if __name__== "__main__":
    main()
