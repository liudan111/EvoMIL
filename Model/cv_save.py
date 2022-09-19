"""Pytorch dataset generate datasets by 5-fold cross validation."""
from collections import Counter
import numpy as np
import pandas as pd
from random import seed
from random import randint
from random import sample      
from sklearn.model_selection import GridSearchCV, train_test_split
import mil_pytorch.mil as mil
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import torch
import time
import os.path
from sklearn.model_selection import KFold
class Dataload:
    def __init__(self, fileindex=0):
        self.fileindex = fileindex
    def create_bags(self):
            # Eukaryota/Prokaryote
            # input_path="esm1b_outputs/Prokaryote/new/"
            input_path="esm1b_outputs/Prokaryote/new1/"
            output_path="esm1b_outputs/Prokaryote/new1/5fold_cv/"
            data = pd.read_csv(input_path + 'data_'+ str(self.fileindex+1) + '.csv', header = None).values
            ids = pd.read_csv(input_path + 'ids_'+ str(self.fileindex+1) + '.csv', squeeze = True, header = None).values
            labels = pd.read_csv(input_path + 'labels_'+ str(self.fileindex+1) + '.csv', squeeze = True, header = None).values
            data = data.astype(np.float32) 
            # Create tensors containing data
            data = torch.tensor(data)
            ids = torch.tensor(ids)
            labels = torch.tensor(labels)
            # Create instance of MilDataset
            dataset = mil.MilDataset(data, ids, labels, normalize = True)
            # Create train and test data loaders (instances of DataLoader)
            batch_size = 1
            indices = np.arange(len(dataset))
   
            train_indices, test_indices = model_selection.train_test_split(indices,test_size = 0.2,shuffle=True,random_state=8080)
        
            test_sampler = SubsetRandomSampler(test_indices)
            test_loader = DataLoader(dataset, sampler = test_sampler, batch_size = batch_size, collate_fn=mil.collate) # test_loader had been fixed

            skf=StratifiedKFold(n_splits=5, random_state=8080, shuffle=True)
            labels_list=labels[train_indices]
       
            r=0
            train_dl_list=[]
            val_dl_list=[]
            for train_index, val_index in skf.split(train_indices,labels_list):
                #   print(labels.np()[train_indices], labels.np()[test_indices])
                  train_indices_real=train_indices[train_index]
                  val_indices_real=train_indices[val_index]
                  train_sampler = SubsetRandomSampler(train_indices_real)
                  val_sampler = SubsetRandomSampler(val_indices_real)
                  train_dl = DataLoader(dataset, sampler = train_sampler, batch_size = batch_size, collate_fn=mil.collate) # Using custom collate_fn mil.collate
                  val_dl = DataLoader(dataset, sampler = val_sampler, batch_size = batch_size, collate_fn=mil.collate)
                  torch.save(train_dl, output_path+'train_dl_'+str(self.fileindex+1)+'_'+str(r))
                  torch.save(val_dl, output_path+'val_dl_'+str(self.fileindex+1)+'_'+str(r))
                  r=r+1
                  train_dl_list.append(train_indices_real)
                  val_dl_list.append(val_indices_real)

if __name__ == "__main__":
    for i in range(21):
        load=Dataload(i)
        load.create_bags()
