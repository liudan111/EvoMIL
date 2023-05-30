from __future__ import print_function
import numpy as np
from sklearn.metrics import auc
from sklearn import metrics
import csv
import os

def aucscore(pred, labels):
    fpr, tpr, threshold = metrics.roc_curve(labels, pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    return auc_score

def get_confusion(pred, target):
    FN = 0
    FP = 0
    TP = 0
    TN = 0
    for yp, y in zip(pred, target):
        if yp == False  and y == True:
            FN += 1
        elif yp == True and y == False:
            FP += 1
        elif yp == True and y == True:
            TP += 1
        else:
            TN += 1
    spec = round(TN / (TN + FP), 6) if (TN + TP) else 'NA'
    sens = round(TP / (TP + FN), 6) if (TP + FN) else 'NA'
    acc  = round((TP + TN) / (len(pred)), 6)
    return acc,spec,sens

#   test  
def results2CSV_test(results,hostname,csvfile):
    results['Hostname']=hostname
    if os.path.isfile(csvfile):
        with open(csvfile, 'a') as csvfile:
            fieldnames = ['Hostname','AUC','Accuracy','f1','spec','sens','prec']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(results) 
    else:
        with open(csvfile, 'a') as csvfile:
            print ( 'new file',csvfile)
            fieldnames = ['Hostname','AUC','Accuracy','f1','spec','sens','prec']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(results) 
            
# train     
def results2CSV_train(results,hostname,csvfile):
    results['Hostname']=hostname
    if os.path.isfile(csvfile):
        with open(csvfile, 'a') as csvfile:
            fieldnames = ['Hostname','train_Loss','train_error','train_acc','Val_Loss','Val_error','Val_Accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(results) 
    else:
        with open(csvfile, 'a') as csvfile:
            print ( 'new file',csvfile)
            fieldnames = ['Hostname','train_Loss','train_error','train_acc','Val_Loss','Val_error','Val_Accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(results)
    