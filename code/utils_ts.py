import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def fun_df_cumsum(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Convert interval-wise time series data to cumulative time series data.
        Args:
            df: interval-wise time series data   
        return:
            df_new: cumulative time series data    
    '''
    fields = list(df)
    total_period = sum(df.ET-df.ST)  ## total trace period as latency
    task_weight = (df.ET-df.ST)/total_period
    task_weight = np.diag(task_weight.values)
    df_new = task_weight.dot(df)
    df_new = pd.DataFrame(data=df_new, columns=fields)
    return df_new.cumsum(), total_period

def fun_cum_vec(v: np.array)-> np.array:
    '''
    Get cumulated columns for plotting CDF
    '''
    vid = np.argmax(v==1)
    if np.sum(v)!=0:
        v[vid:]=1
    else:
        v[-1]=1
    return v

def get_TPR_FPR(pred, alpha, n_stra):
    '''
    Get true positive and false positive rate from prediction matrix/list
    pred: prediction. 
    alpha: threshold.
    n_stra: number of true stragglers.
    '''
    aa = np.array(pred)
    ppdd1 = np.apply_along_axis(max, 0, aa)
    aa = aa >= alpha
    aa = aa.astype(int) 
    aa_cum = np.apply_along_axis(fun_cum_vec, 0, aa)
    stra_cum = aa_cum[:, :n_stra]
    gap_cum = aa_cum[:, n_stra:]
    tpr = sum(stra_cum[-2])*100/len(stra_cum[-2])
    fpr = sum(gap_cum[-2])*100/len(gap_cum[-2])
    
    true = np.asarray([1]*len(stra_cum[-2]) + [0] * len(gap_cum[-2]))
    auc = roc_auc_score(true, ppdd1)
    
    ppdd2 = np.concatenate([stra_cum[-2], gap_cum[-2]])    
    f1 = f1_score(true, ppdd2)
    
    tn, fp, fn, tp = confusion_matrix(true, ppdd2).ravel()
    fnr = fn/(fn+tp)
    
    return tpr, fpr, fnr, auc, f1

def get_PCT_new(kl_pred: list, y_true: np.array, n_stra:int, alpha: float, bins: list) -> list:
    """Get recall between prediction and groundtruth

    Args:    
      y_true: groundtruth real value array.
      n_stra: number of true stragglers.
      kl_pred: prediction matrix.
      bins: bin size interval.

    Returns:
      recall: recall for [90,95), [95, 99), [99+]
    """
    aa = np.array(kl_pred)[:,:n_stra]
    aa = aa >= alpha
    aa = aa.astype(int)
    aa_cum = np.apply_along_axis(fun_cum_vec, 0, aa)
    y_pred_b = aa_cum[-2]
    y_pos_sort = np.argsort(y_true)
    s95  = sum(y_pred_b[y_pos_sort[0:bins[0]].tolist()])*100/(bins[0])
    s99  = sum(y_pred_b[y_pos_sort[bins[0]:bins[1]].tolist()])*100/(bins[1]-bins[0])
    s99p = sum(y_pred_b[y_pos_sort[bins[1]:bins[2]].tolist()])*100/(bins[2]-bins[1])

    return [s95, s99, s99p]

def get_PCT(kl_pred: list, y_true: np.array, alpha: float, bins: list) -> list:
    """Get recall between prediction and groundtruth

    Args:    
      y_true: groundtruth real value array.
      kl_pred: prediction matrix.
      bins: bin size interval.

    Returns:
      recall: recall for [90,95), [95, 99), [99+]
    """
    aa = np.array(kl_pred)
    aa = aa >= alpha
    aa = aa.astype(int)
    aa_cum = np.apply_along_axis(fun_cum_vec, 0, aa)
    y_pred_b = aa_cum[-2]
    y_pos_sort = np.argsort(y_true)
    s95  = sum(y_pred_b[y_pos_sort[0:bins[0]].tolist()])*100/(bins[0])
    s99  = sum(y_pred_b[y_pos_sort[bins[0]:bins[1]].tolist()])*100/(bins[1]-bins[0])
    s99p = sum(y_pred_b[y_pos_sort[bins[1]:bins[2]].tolist()])*100/(bins[2]-bins[1])

    return [s95, s99, s99p]

def get_FPR(pred: list, alpha: float) -> float:
    '''
    Get false positive rate from prediction matrix/list
    pred: prediction 
    alpha: threshold
    '''
    pred_b = [1 if i >= alpha else 0 for i in pred]
    return sum(pred_b)*100/len(pred)


def get_TPR(kl_pred: list, alpha: float) -> float: 
    '''
    Get true positive rate from prediction matrix/list
    kl_input: input prediction list
    alpha: threshold
    '''
    aa = np.array(kl_pred)
    aa = aa >= alpha
    aa = aa.astype(int)
    aa_cum = np.apply_along_axis(fun_cum_vec, 0, aa)
    return sum(aa_cum[-2])*100/len(aa_cum[-2])


def fun_cum_tpr(kl_input: list, alpha: float) -> np.array:
    '''
    Get cumulative probability from result matrix/list
    kl_input: input result list
    alpha: threshold
    tl_list: true task length for different tasks
    '''
    aa=np.array(kl_input)
    aa = aa >= alpha
    aa = aa.astype(int)
    aa_cum = np.apply_along_axis(fun_cum_vec, 0, aa)
    aa_cum_pr = np.sum(aa_cum, axis=1)/aa.shape[1]
    # return aa_cum_pr * 100
    return np.insert(aa_cum_pr,0,0)*100.0


