import numpy as np
import pickle
from collections import defaultdict
from multiprocessing import Process, Pool
import sys
import os
from os import path
import csv
import pandas as pd


indirectory = './alibabatimeseries2017/'

def filter(file):
    if file_len(indirectory+file) < 100:
        os.remove(indirectory+file)
    
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    f.close()
    return i + 1

def main():
    for filename in os.listdir(indirectory):
        filter(filename)
        
if __name__== "__main__":
    main()
