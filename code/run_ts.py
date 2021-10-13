import os
import numpy as np
import pandas as pd
import random
import time
import sys
import argparse

from os import listdir
from os.path import isfile, join

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor

from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler
from pulearn import ElkanotoPuClassifier
from pulearn import BaggingPuClassifier
from tobit import *
import KTBoost.KTBoost as KTBoost

from utils_ts import *

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
  parser.add_argument('--gamma', type=float, default=0, help='Parameter for propensity score adjustment (default: 0)')
  parser.add_argument('--out', type=str, default='out', help='Output folder to save results (default: out)')

  args = parser.parse_args()
 
  path_ts   = args.data_path  
  jobid     = args.jobid
  gamma     = args.gamma  # Parameter to tune propensity score
  pt        = args.pt     # Training set size
  tail      = args.tail   # Latency threshold
  rs        = args.rs     # Random state
  out       = args.out    # Output folder to save results

  if not os.path.exists(out):
    sys.exit("No Result Folder Created!")

  print("data_path: {}".format(path_ts))
  print("jobid:     {}".format(jobid))
  print("gamma:     {}".format(gamma))
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

  X_train_up, X_test_up, Y_train_up, Y_test_up = X_train, X_test_final, Y_train, Y_test_final
  Y_train_pu = (Y_train_up<alpha)*1

  lt_stra, lt_gap = len(list_task_final_stra), len(list_task_final_gap)  ## straggler/non-straggler size in testing
  list_task_final_gap_down = list_task_final_gap
  list_task_final_test_down = list_task_final_test
  full_idx = range(len(list_task_final_test_down))
  eps = 0.05
  pl_gb, pl_ipwnc, pl_ipw, pl_en, pl_bg, pl_tb, pl_kt = [],[],[],[],[],[],[]

  for k in range(2, ts_init_size):  # ts_init_size
      
      tn = [i.iloc[k].values for i in list_task_final_test_down]
      np_tn = np.asarray(tn)    
      np_tn_nzidx = (np.where(np_tn.any(axis=1))[0]).tolist()
      np_tn_nz = np_tn[~np.all(np_tn == 0, axis=1)]
      
      cen_train = np.mean(X_train_up, axis=0)
      cen_test = np.mean(np_tn_nz, axis=0)
      rho = sum(cen_train**2)/sum((cen_train-cen_test)**2)
      
      if k == 2:
        delta = 1/(1+rho) - gamma
        print("delta: {}".format(delta))
      
      ## Base      
      start_time = time.time()
      r_gb = GradientBoostingRegressor(random_state=rs).fit(X_train_up, Y_train_up)
      p_gb_curr = r_gb.predict(np_tn_nz).tolist()
      tm_gb = time.time() - start_time
      p_gb = [0] * len(full_idx)
      for j in range(len(np_tn_nzidx)):
          p_gb[np_tn_nzidx[j]] = p_gb_curr[j]
      pl_gb.append(p_gb)        
      
      ## IPW-NC
      X = np.asarray(np.concatenate((X_train_up, np_tn_nz)))
      y = np.asarray([0] * X_train_up.shape[0] + [1] * np_tn_nz.shape[0])
      clf = LogisticRegression(random_state=rs, solver='lbfgs').fit(X, y)
      ps = clf.predict_proba(X)
      tm_ipwnc = time.time() - start_time
      tm_ipw = time.time() - start_time
      ps0 = ps[X_train_up.shape[0]:,0].copy()
      p_ipwnc_curr = [x/z for x, z in zip(p_gb_curr, ps0.tolist())]
      p_ipwnc = [0] * len(full_idx)
      for j in range(len(np_tn_nzidx)):
          p_ipwnc[np_tn_nzidx[j]] = p_ipwnc_curr[j]    
      pl_ipwnc.append(p_ipwnc)      
      
      ## IPW
      ps1 = ps[X_train_up.shape[0]:,0].copy()    
      for i in range(len(ps1)):
          ps1[i] = max(eps, min(ps1[i]+delta, 1))     
              
      p_ipw_curr = [x/z for x, z in zip(p_gb_curr, ps1.tolist())]
      p_ipw = [0] * len(full_idx)
      for j in range(len(np_tn_nzidx)):
          p_ipw[np_tn_nzidx[j]] = p_ipw_curr[j]    
      pl_ipw.append(p_ipw)     
      
      ## PU
      Xp_train_up = np.concatenate((X_train_up, np_tn_nz))
      Yp_train_up = np.asarray([1]*X_train_up.shape[0]+[0]*len(np_tn_nz)) 
      try:    
          start_time = time.time()
          r_en = ElkanotoPuClassifier(estimator=SVC(probability=True),hold_out_ratio=0.1).fit(Xp_train_up, Yp_train_up)
          p_en_curr = (0.5-r_en.predict(np_tn_nz)/2).tolist()
          tm_en = time.time() - start_time
          start_time = time.time()
          r_bg = BaggingPuClassifier(base_estimator=SVC(probability=True)).fit(Xp_train_up, Yp_train_up)
          p_bg_curr = (1-r_bg.predict(np_tn_nz)).tolist() 
          tm_bg = time.time() - start_time
      except:
          r_bg = BaggingPuClassifier(base_estimator=SVC(probability=True)).fit(Xp_train_up, Yp_train_up)
          r_en = r_bg
          p_en_curr = (0.5-r_en.predict(np_tn_nz)/2).tolist()
          p_bg_curr = (1-r_bg.predict(np_tn_nz)).tolist() 
      
      p_en = [0] * len(full_idx)
      for j in range(len(np_tn_nzidx)):
          p_en[np_tn_nzidx[j]] = p_en_curr[j]    
      pl_en.append(p_en)         
          
      p_bg = [0] * len(full_idx)
      for j in range(len(np_tn_nzidx)):
          p_bg[np_tn_nzidx[j]] = p_bg_curr[j]    
      pl_bg.append(p_bg)     
      
      ## Tobit
      Xt_train_up = np.concatenate((X_train_up, np_tn_nz))
      Yt_train_up = np.concatenate((Y_train_up, Y_test_up[np_tn_nzidx]))   
      Xt_train_up = pd.DataFrame(Xt_train_up)
      Yt_train_up = pd.Series(Yt_train_up)
      upper = max(Y_train_up)
      right = Yt_train_up > upper
      Yt_train_up = Yt_train_up.clip(upper=upper)              
      cens = pd.Series(np.zeros((Xt_train_up.shape[0],)))
      cens[right] = 1 
      start_time = time.time()
      p_tb_curr = TobitModel().fit(Xt_train_up,Yt_train_up,cens,verbose=False).predict(pd.DataFrame(np_tn_nz)).tolist()
      tm_tb = time.time() - start_time
      p_tb = [0] * len(full_idx)
      for j in range(len(np_tn_nzidx)):
          p_tb[np_tn_nzidx[j]] = p_tb_curr[j]    
      pl_tb.append(p_tb)    
      
      
      ## Grabit
      ## Get variance         
      init_reg = LinearRegression(fit_intercept=False).fit(Xt_train_up, Yt_train_up)
      y_pred = init_reg.predict(Xt_train_up)
      sigma0 = np.sqrt(np.var(Yt_train_up - y_pred))
      start_time = time.time()
      kt = KTBoost.BoostingRegressor(loss='tobit', yl=0, yu=upper, sigma=sigma0).fit(Xt_train_up, Yt_train_up)
      p_kt_curr = kt.predict(np_tn_nz).tolist()
      tm_kt = time.time() - start_time
      p_kt = [0] * len(full_idx)
      for j in range(len(np_tn_nzidx)):
          p_kt[np_tn_nzidx[j]] = p_kt_curr[j]         
      pl_kt.append(p_kt)     
      
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

  TPR_gb, FPR_gb, FNR_gb, AUC_gb, F1_gb = get_TPR_FPR(pl_gb, alpha, num_stra)
  TPR_ipwnc, FPR_ipwnc, FNR_ipwnc, AUC_ipwnc, F1_ipwnc = get_TPR_FPR(pl_ipwnc, alpha, num_stra)
  TPR_ipw, FPR_ipw, FNR_ipw, AUC_ipw, F1_ipw = get_TPR_FPR(pl_ipw, alpha, num_stra)
  TPR_en, FPR_en, FNR_en, AUC_en, F1_en = get_TPR_FPR(pl_en, alpha, num_stra)
  TPR_bg, FPR_bg, FNR_bg, AUC_bg, F1_bg = get_TPR_FPR(pl_bg, alpha, num_stra)
  TPR_tb, FPR_tb, FNR_tb, AUC_tb, F1_tb = get_TPR_FPR(pl_tb, alpha, num_stra)
  TPR_kt, FPR_kt, FNR_kt, AUC_kt, F1_kt = get_TPR_FPR(pl_kt, alpha, num_stra)

  TPR_L = [TPR_gb, TPR_ipwnc, TPR_ipw, TPR_en, TPR_bg, TPR_tb, TPR_kt]
  FPR_L = [FPR_gb, FPR_ipwnc, FPR_ipw, FPR_en, FPR_bg, FPR_tb, FPR_kt]
  FNR_L = [FNR_gb, FNR_ipwnc, FNR_ipw, FNR_en, FNR_bg, FNR_tb, FNR_kt]
  AUC_L = [AUC_gb, AUC_ipwnc, AUC_ipw, AUC_en, AUC_bg, AUC_tb, AUC_kt]
  F1_L = [F1_gb, F1_ipwnc, F1_ipw, F1_en, F1_bg, F1_tb, F1_kt]

  df_acc = pd.DataFrame(list(zip(TPR_L,FPR_L,FNR_L,AUC_L,F1_L)), columns=['TPR', 'FPR', 'FNR', 'AUC', 'F1'],
                       index=['gb', 'ipwnc', 'ipw', 'en', 'bg', 'tb', 'kt'])
  # df_acc.to_csv('res_ts/acc/Job{}_acc.csv'.format(jobid))
  df_acc

  ## Get percentile results
  PCT_gb = get_PCT_new(pl_gb, Y_test_stra_final, num_stra, alpha, BI_new)
  PCT_ipwnc = get_PCT_new(pl_ipwnc, Y_test_stra_final, num_stra, alpha, BI_new)
  PCT_ipw = get_PCT_new(pl_ipw, Y_test_stra_final, num_stra, alpha, BI_new)
  PCT_en = get_PCT_new(pl_en, Y_test_stra_final, num_stra, alpha, BI_new)
  PCT_bg = get_PCT_new(pl_bg, Y_test_stra_final, num_stra, alpha, BI_new)
  PCT_tb = get_PCT_new(pl_tb, Y_test_stra_final, num_stra, alpha, BI_new)
  PCT_kt = get_PCT_new(pl_kt, Y_test_stra_final, num_stra, alpha, BI_new)
 
  np_pct = np.concatenate([np.asarray(PCT_gb).reshape(1,-1),np.asarray(PCT_ipwnc).reshape(1,-1),
                          np.asarray(PCT_ipw).reshape(1,-1),
                          np.asarray(PCT_en).reshape(1,-1),np.asarray(PCT_bg).reshape(1,-1),
                          np.asarray(PCT_tb).reshape(1,-1), np.asarray(PCT_kt).reshape(1,-1)])
  df_pct = pd.DataFrame(np_pct, columns=['<95','<99','99+'], 
                        index=['gb', 'ipwnc', 'ipw', 'en', 'bg', 'tb', 'kt'])
  df_pct.fillna(100, inplace=True)
  # df_pct.to_csv('res_ts/ptc/Job{}_ptc.csv'.format(jobid))
  df_pct

if __name__ == '__main__':
  main()














