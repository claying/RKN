import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score,
    precision_recall_curve, average_precision_score
)


def bootstrap_auc(y_true, y_pred, ntrial=10):
    n = len(y_true)
    aucs = []
    for t in range(ntrial):
        sample = np.random.randint(0, n, n)
        try:
            auc = roc_auc_score(y_true[sample], y_pred[sample])
        except:
            return np.nan, np.nan  # If any of the samples returned NaN, the whole bootstrap result is NaN
        aucs.append(auc)
    return np.mean(aucs), np.std(aucs)


def recall_at_fdr(y_true, y_score, fdr_cutoff=0.05):
    # print y_true, y_score
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    fdr = 1 - precision
    cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
    return recall[cutoff_index]


def xroc(res, cutoff):
    """
    :type res: List[List[label, score]]
    :type curoff: all or 50
    """
    area, height, fp, tp = 0.0, 0.0, 0.0, 0.0
    for x in res:
        label = x
        if cutoff > fp:
            if label == 1:
                height += 1
                tp += 1
            else:
                area += height
                fp += 1
        else:
            if label == 1:
                tp += 1
    lroc = 0
    if fp != 0 and tp != 0:
        lroc = area / (fp * tp)
    elif fp == 0 and tp != 0:
        lroc = 1
    elif fp != 0 and tp == 0:
        lroc = 0
    return lroc


def get_roc(y_true, y_pred, cutoff):
    score = []
    label = []

    for i in range(y_pred.shape[0]):
        label.append(y_true[i])
        score.append(y_pred[i])

    index = np.argsort(score)
    index = index[::-1]
    t_score = []
    t_label = []
    for i in index:
        t_score.append(score[i])
        t_label.append(label[i])

    score = xroc(t_label, cutoff)
    return score


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    metric = {}
    metric['log.loss'] = log_loss(y_true, y_pred)
    metric['accuracy'] = accuracy_score(y_true, y_pred > 0.5)
    metric['auROC'] = roc_auc_score(y_true, y_pred)
    metric['auROC50'] = get_roc(y_true, y_pred, 50)
    metric['auPRC'] = average_precision_score(y_true, y_pred)
    metric['recall_at_10_fdr'] = recall_at_fdr(y_true, y_pred, 0.10)
    metric['recall_at_5_fdr'] = recall_at_fdr(y_true, y_pred, 0.05)
    metric["pearson.r"], metric["pearson.p"] = stats.pearsonr(y_true, y_pred)
    metric["spearman.r"], metric["spearman.p"] = stats.spearmanr(y_true, y_pred)
    df = pd.DataFrame.from_dict(metric, orient='index')
    df.columns = ['value']
    df.sort_index(inplace=True)
    return df
