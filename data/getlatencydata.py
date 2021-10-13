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

with open(basedirectory+taskstartoutfile, 'rb') as handle:
        tasksstartdictionary = pickle.load(handle)
with open(basedirectory+tasksendoutfile, 'rb') as handle:
        tasksenddictionary = pickle.load(handle)

count = 0
for (task in tasksstartdictionary):
    if task in tasksenddictionary:
        if tasksenddictionary[task] - tasksstartdictionary[task] > 300000000:
            count = count + 1
print("large enough" + count + "\n")

