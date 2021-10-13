import os
import numpy as np
import pandas as pd
import random
import time
import sys
import argparse

from os import listdir
from os.path import isfile, join

# Import all OD models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sos import SOS
from pyod.models.lscp import LSCP
from pyod.models.cof import COF
from pyod.models.sod import SOD
from pyod.models.xgbod import XGBOD


def main():

  #####################################################
  ########## Read data and simple processing ########## 
  #####################################################

  parser = argparse.ArgumentParser(description='Straggle Prediction on Live Data.')
  parser.add_argument('--data_path', type=str, help='Data path')
  parser.add_argument('--jobid', type=str, default="6343048076", help='Job ID') 
  parser.add_argument('--rs', type=int, default=42, help='Random state (default: 42)')
  parser.add_argument('--pt', type=float, default=0.04, help='Training set size (default: 0.2)')
  parser.add_argument('--tail', type=float, default=0.9, help='Latency threshold (default: 0.9)')
  parser.add_argument('--delta', type=float, default=0, help='Parameter for propensity score adjustment (default: 0)')
  parser.add_argument('--out', type=str, default='out', help='Output folder to save results (default: out)')

  args = parser.parse_args()
 
  path_ts   = args.data_path  
  jobid     = args.jobid
  delta     = args.delta  # Parameter to tune propensity score
  pt        = args.pt     # Training set size
  tail      = args.tail   # Latency threshold
  rs        = args.rs     # Random state
  out       = args.out    # Output folder to save results

  if not os.path.exists(out):
    sys.exit("No Result Folder Created!")

  print("data_path: {}".format(path_ts))
  print("jobid:     {}".format(jobid))
  print("delta:     {}".format(delta))
  print("pt   :     {}".format(pt))
  print("tail:      {}".format(tail))
  print("rs   :     {}".format(rs))
  print("out :      {}".format(out))

  path_ts_file = path_ts + jobid
  files_task = [f for f in listdir(path_ts_file) if isfile(join(path_ts_file, f))]
  job_rawsize = len(files_task)  ## Get number of tasks in a job

  task_colnames = ['ST','ET','JI', 'TI','MI', 'MCU', 'CMU', 'AMU', 'UPC', 'TPC', 'MAXMU', 'MIO',
            'MDK', 'MAXCPU', 'MAXIO', 'CPI', 'MAI', 'SP', 'AT', 'SCPU', 'EV', 'FL']
  task_fields = ['ST','ET','MCU', 'CMU', 'AMU', 'UPC', 'TPC', 'MAXMU', 'MIO',
            'MDK', 'MAXCPU', 'MAXIO', 'CPI', 'MAI', 'SCPU', 'EV', 'FL']
  task_cols = ['Latency','MCU', 'CMU', 'AMU', 'UPC', 'TPC', 'MAXMU', 'MIO',
            'MDK', 'MAXCPU', 'MAXIO', 'CPI', 'MAI', 'SCPU', 'EV', 'FL']

  ## Get cumulative time series data
  list_task = [] 
  list_tp = []  ## list of total period
  list_task_compact = []  ## list of last row
  for i in range(job_rawsize):
    task = pd.read_csv('{}/{}/{}'.format(path_ts,jobid,i), header=None,
                       names=task_colnames, usecols=task_fields)
    task_new, tp_new = fun_df_cumsum(task)
    list_tp.append(tp_new)
    list_task_compact.append(task_new.iloc[-1].tolist())
    list_task.append(task_new)

  ## Construct new non-time series data
  np_task_compact = np.array(list_task_compact)
  df_task_compact = pd.DataFrame(np_task_compact, columns=task_fields)
  df_task_compact['Latency'] = pd.Series(np.asarray(list_tp), index=df_task_compact.index)
  df_sel = df_task_compact[task_cols]
  job = (df_sel-df_sel.min())/(df_sel.max()-df_sel.min())
  job = job.dropna(axis='columns') 
  job_raw = job.reset_index(drop=True)

  ## Normalize task at different time points using final row
  list_task_nn = []
  ts_size = 0  ## max task size in a job
  cn_train = [i for i in list(job) if i not in ['Latency']]  
  for i in range(len(list_task)):
      task = list_task[i][cn_train]
      task = (task-df_sel[cn_train].min())/(df_sel[cn_train].max()-df_sel[cn_train].min())
      if ts_size < task.shape[0]:
          ts_size = task.shape[0]
      list_task_nn.append(task)

  #####################################################################################
  ########## Now we have complete job data constructed from time series data ########## 
  #####################################################################################

  ## Split training and testing data
  latency = job_raw.Latency.values
  ## Parameter to tune propensity score
  lat_sort = np.sort(latency)
  tail = 0.9

  print("# tail :  {}".format(tail))

  cutoff = int(tail*latency.shape[0])
  print("# cutoff:  {}".format(cutoff))
  alpha = lat_sort.tolist()[cutoff]
  print("# alpha:  {}".format(alpha))

  pt = 0.04
  cutoff_pt = int(pt * latency.shape[0])
  alpha_pt = lat_sort.tolist()[cutoff_pt]
  train_idx_init = job.index[job['Latency'] < alpha].tolist()
  test_idx_init = job.index[job['Latency'] >= alpha].tolist()
  train_idx_removed = job.index[(job['Latency'] >= alpha_pt) & (job['Latency'] < alpha)].tolist()
  print("# true tail: {}".format(len(test_idx_init)))

  train_idx = list(set(train_idx_init) - set(train_idx_removed))
  test_idx = test_idx_init + train_idx_removed  ## test_idx = stra_idx + gap_idx
  print("# removed: {}".format(len(train_idx_removed)))

  job =job_raw.copy() 
  job_train = job.iloc[train_idx]
  job_test = job.iloc[test_idx]
  job_test_stra = job.iloc[test_idx_init]
  job_test_gap = job.iloc[train_idx_removed]
  print("# train: {}".format(job_train.shape[0]))
  print("# test:  {}".format(job_test.shape[0]))
  print("# test stra:  {}".format(job_test_stra.shape[0]))
  print("# test gap:  {}".format(job_test_gap.shape[0]))

  X_train = job_train.to_numpy()[:,1:]
  Y_train = job_train.to_numpy()[:,0]
  X_test = job_test.to_numpy()[:,1:]
  Y_test = job_test.to_numpy()[:,0]
  X_test_stra = job_test_stra.to_numpy()[:,1:]
  Y_test_stra = job_test_stra.to_numpy()[:,0]
  X_test_gap = job_test_gap.to_numpy()[:,1:]
  Y_test_gap = job_test_gap.to_numpy()[:,0]

  job.loc[train_idx_init, 'Label'] = 0
  job.loc[test_idx_init, 'Label'] = 1
  y_test_true = job.loc[test_idx, 'Label'].values ## binary groundtruth for testing tasks
  y_stra_true = job.loc[test_idx_init, 'Latency'].values ## groundtruth for straggler

  ## Get latency bins, [90,95), [95, 99), [99+]
  cutoff95 = int(0.95 * latency.shape[0])
  alpha95 = lat_sort.tolist()[cutoff95]
  cutoff99 = int(0.99 * latency.shape[0])
  alpha99 = lat_sort.tolist()[cutoff99]
  test95_idx = job.index[(job['Latency'] >= alpha) & (job['Latency'] < alpha95)].tolist()
  test99_idx = job.index[(job['Latency'] >= alpha95) & (job['Latency'] < alpha99)].tolist()
  test99p_idx = job.index[(job['Latency'] >= alpha99)].tolist()
  BI = np.cumsum([len(test95_idx), len(test99_idx), len(test99p_idx)])
  print("# latency bins: {}".format(BI))

  ###################################################
  ########## Start time series experiments ########## 
  ###################################################

  ## Padding zero rows to unify task size
  list_task_norm = []
  test_idx_gap = [i for i in test_idx if i not in test_idx_init]
  list_task_nn_stra = [list_task_nn[i] for i in test_idx_init]  ## only straggler tasks
  list_task_nn_gap = [list_task_nn[i] for i in test_idx_gap] ## nonstragglers in testing
  list_task_nn_test = [list_task_nn[i] for i in test_idx]  ## for all test tasks

  ss_stra, ss_gap = [d.shape[0] for d in list_task_nn_stra], [d.shape[0] for d in list_task_nn_gap]
  ts_init_size = np.max(ss_stra)   ## max task size/time intervals for stragglers
  print(ts_init_size) 

  for dd in list_task_nn:
      if dd.shape[0] < ts_init_size:
          df2 =  pd.DataFrame(np.zeros([(ts_init_size-dd.shape[0]),dd.shape[1]]), columns=list(dd))
          list_task_norm.append(dd.append(df2, ignore_index=True))
      else:
          list_task_norm.append(dd)       
                 
  ## Only care about tasks that are stragglers
  list_task_norm_stra = [list_task_norm[i] for i in test_idx_init]
  list_task_norm_gap = [list_task_norm[i] for i in test_idx_gap]
  list_task_norm_test = [list_task_norm[i] for i in test_idx]
  print(len(list_task_norm_stra),len(list_task_norm_gap),len(list_task_norm_test))

  ## Process task length
  mean_len_stra = sum([len(i) for i in list_task_nn_stra])/len(list_task_nn_stra)
  mean_len_gap = sum([len(i) for i in list_task_nn_gap])/len(list_task_nn_gap)
  print(mean_len_stra, mean_len_gap)

  ## Remove bad true straggler
  bad_stra_idx = []
  for i in range(len(list_task_nn_stra)):
      if len(list_task_nn_stra[i]) < mean_len_stra-1:
          bad_stra_idx.append(i)
  good_stra_idx = [i for i in range(len(list_task_nn_stra)) if i not in bad_stra_idx] 
  list_task_final_stra = [list_task_norm_stra[i] for i in good_stra_idx]

  ## Remove bad normal tasks
  bad_gap_idx = []
  for i in range(len(list_task_nn_gap)):
      if len(list_task_nn_gap[i]) > mean_len_gap+1:
          bad_gap_idx.append(i)
  good_gap_idx = [i for i in range(len(list_task_nn_gap)) if i not in bad_gap_idx]  
  list_task_final_gap = [list_task_norm_gap[i] for i in good_gap_idx]

  list_task_final_test = list_task_final_stra + list_task_final_gap

  ## Get final test data
  X_test_stra_final = X_test_stra[good_stra_idx]
  X_test_gap_final = X_test_gap[good_gap_idx]
  X_test_final = np.concatenate((X_test_stra_final, X_test_gap_final))

  Y_test_stra_final = Y_test_stra[good_stra_idx]
  Y_test_gap_final = Y_test_gap[good_gap_idx]
  Y_test_final = np.concatenate((Y_test_stra_final, Y_test_gap_final))

  BI_95 = sum(((Y_test_stra_final>=alpha) & (Y_test_stra_final < alpha95)) * 1)
  BI_99 = sum(((Y_test_stra_final>=alpha) & (Y_test_stra_final < alpha99)) * 1)
  BI_99p = sum(((Y_test_stra_final>=alpha)) * 1)
  BI_new = np.asarray([BI_95, BI_99, BI_99p])

  num_stra, num_gap = X_test_stra_final.shape[0], X_test_gap_final.shape
  print(X_test_stra_final.shape, X_test_gap_final.shape, X_test_final.shape)
  print(BI_new)


  ###################################################
  ##################Online training##################
  ###################################################

  # random_state = 42
  outliers_fraction = 0.1
  # initialize a set of detectors for LSCP
  detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10)]

  # Define nine outlier detection tools to be compared
  classifiers = {
      'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
      'Cluster-based Local Outlier Factor (CBLOF)': CBLOF(contamination=outliers_fraction, check_estimator=False, random_state=rs),
      'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
      'Isolation Forest': IForest(contamination=outliers_fraction, random_state=rs),
      'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
      'Local Outlier Factor (LOF)': LOF(n_neighbors=15, contamination=outliers_fraction),
      'Minimum Covariance Determinant (MCD)': MCD(contamination=outliers_fraction, random_state=rs),
      'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
      'Principal Component Analysis (PCA)': PCA( contamination=outliers_fraction, random_state=rs),
      'Stochastic Outlier Selection (SOS)': SOS(contamination=outliers_fraction),
      'Locally Selective Combination (LSCP)': LSCP(detector_list, contamination=outliers_fraction, random_state=rs),
      'Connectivity-Based Outlier Factor (COF)':COF(n_neighbors=15, contamination=outliers_fraction),
      'Subspace Outlier Detection (SOD)': SOD(contamination=outliers_fraction),
      'Supervised Outlier Detection (XGBOD)': XGBOD(contamination=outliers_fraction)
  }


  clf_list = ['abod', 'cblof', 'hbos', 'iforest', 'knn', 'lof', 'mcd', 'ocsvm', 'pca', 
              'sos', 'lscp', 'cof', 'sod','xgbod']
  pl = {'abod':[], 'cblof':[], 'hbos':[], 'iforest':[], 'knn': [], 'lof':[], 'mcd':[], 
        'ocsvm':[], 'pca':[], 'sos':[], 'lscp':[], 'cof':[], 'sod':[], 'xgbod':[]}

  X_train_up, X_test_up, Y_train_up, Y_test_up = X_train, X_test_final, Y_train, Y_test_final
  Y_train_pu = (Y_train_up<alpha)*1

  lt_stra, lt_gap = len(list_task_final_stra), len(list_task_final_gap)  ## straggler/non-straggler size in testing
  list_task_final_gap_down = list_task_final_gap
  list_task_final_test_down = list_task_final_test
  full_idx = range(len(list_task_final_test_down))


  for k in range(2, ts_init_size): 

    print('k = {}'.format(k))
    tn = [i.iloc[k].values for i in list_task_final_test_down]
    np_tn = np.asarray(tn)    
    np_tn_nzidx = (np.where(np_tn.any(axis=1))[0]).tolist()
    np_tn_nz = np_tn[~np.all(np_tn == 0, axis=1)]
    
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        print(i + 1, 'fitting', clf_name)
        if i !=13:            
            clf.fit(X_train_up)
        else:
            ## For XGBOD, supervised
            bb = Y_train_up.tolist()
            Y_train_up_b = [int(b>alpha) for b in bb]
            clf.fit(X_train_up, Y_train_up_b)
        y_pred = clf.predict(np_tn_nz).tolist()
        
        p_curr = [0] * len(full_idx)
        for j in range(len(np_tn_nzidx)):
            p_curr[np_tn_nzidx[j]] = y_pred[j]
        pl[clf_list[i]].append(p_curr)
        
    ## Update training
    zero_idx_now = (np.where(~np_tn.any(axis=1))[0]).tolist() 
    nonzero_idx_now = [i for i in full_idx if i not in zero_idx_now]  
    
    if k < ts_init_size-1:
        tn_next = [i.iloc[k+1].values for i in list_task_final_test]
        np_tn_next = np.asarray(tn_next)
        zero_idx_next = (np.where(~np_tn_next.any(axis=1))[0]).tolist() 
        add_idx = [x for x in nonzero_idx_now if x in zero_idx_next]        
    else:
        add_idx = nonzero_idx_now

    X_train_up = np.concatenate((X_train_up, X_test_up[add_idx]))
    Y_train_up = np.concatenate((Y_train_up, Y_test_up[add_idx]))  
      
  #### Get TPR/FPR results


  TPR_L, FPR_L, FNR_L, AUC_L, F1_L = [], [], [], [], []
  for i, pred in enumerate(pl.items()):
      TPR, FPR, FNR, AUC, F1 = get_TPR_FPR(pl[clf_list[i]], alpha, num_stra)
      TPR_L = TPR_L + [TPR]
      FPR_L = FPR_L + [FPR]
      FNR_L = FNR_L + [FNR]
      AUC_L = AUC_L + [AUC]
      F1_L  = F1_L + [F1]
      
  df_acc = pd.DataFrame(list(zip(TPR_L,FPR_L,FNR_L,AUC_L,F1_L)), columns=['TPR', 'FPR', 'FNR', 'AUC', 'F1'],
                       index=clf_list)
  # df_acc.to_csv('res_ts_od/acc/Job{}_acc.csv'.format(jobid))
  df_acc

  ## Get percentile results
  PCT_L= []
  for i, pred in enumerate(pl.items()):
      PCT = get_PCT_new(pl[clf_list[i]], Y_test_stra_final, num_stra, alpha, BI_new)
      PCT_L  = PCT_L + [PCT]
  ## Get percentile results dataframe
  df_pct = pd.DataFrame(np.asarray(PCT_L), columns=['<95','<99','99+'], index=clf_list)
  df_pct.fillna(100, inplace=True)
  # df_pct.to_csv('res_ts_od/ptc/Job{}_ptc.csv'.format(jobid))
  df_pct 

if __name__ == '__main__':
  main()














