from __future__ import print_function
import torch
from torch.autograd import Variable
from sklearn.metrics import auc
import os
import csv
# test_mc

""" metrics: topks_correct and topk_accuracies. Copyright ©  Facebook.
"""
def topks_correct(preds, labels, ks):
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [
        top_max_k_correct[:k, :].view(-1).float().sum() for k in ks
    ]
    return topks_correct

def topk_accuracies(preds, labels, ks):
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]

# test_mc
def results2CSV_test(results,csvfile):
    if os.path.isfile(csvfile):
        with open(csvfile, 'a') as csvfile:
            fieldnames = ['AUC','Accuracy','f1']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(results) 
    else:
        with open(csvfile, 'a') as csvfile:
            print ( 'new file',csvfile)
            fieldnames = ['AUC','Accuracy','f1']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(results) 

# train_mc
def results2CSV_train(results,csvfile):
    if os.path.isfile(csvfile):
        with open(csvfile, 'a') as csvfile:
            fieldnames = ['train_Loss','train_acc','Val_Loss','Val_Accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(results) 
    else:
        with open(csvfile, 'a') as csvfile:
            print ( 'new file',csvfile)
            fieldnames = ['train_Loss','train_acc','Val_Loss','Val_Accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(results)