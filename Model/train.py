from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from model_esm1b import Attention, GatedAttention
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import pandas as pd
import os
import csv
from sklearn.metrics import f1_score
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\n GPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('Init Model')
if args.model=='attention':
    model = Attention()
elif args.model=='gated_attention':
    model = GatedAttention()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

def accuracy(pred, target, threshold = 0):
    pred = pred.detach().numpy()
    target = target.detach().numpy()
    acc=np.sum(target == pred)/target.shape[0]
    return acc
        
def results2CSV(results,hostname,csvfile):
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
             
if __name__ == "__main__":
    # Eukaryota/Prokaryote
    input='/home1/2656169l/data/Prokaryote/new1/'
    output='/home1/2656169l/data/Prokaryote/new1/5fold_model/'
    # virus_pos_neg_path= input+'virus_' 
    virus_host_30 = pd.read_csv(input+'final_specise_sort_30.csv') #Eukaryota_final_specise_sort_30
    length=virus_host_30.shape[0]
    output_path="esm1b_outputs/Prokaryote/new1/5fold_cv/"
    snapStep=10 #every 10 epochs we will validate our validation set once
 
    for i in range(length):
        hostname=virus_host_30.iloc[i,0]
        for j in range(5):
            print('Start Training')
            max_acc = 0                
            train_loader=torch.load(output_path+'train_dl_'+str(i+1)+'_'+str(j))
            val_loader=torch.load(output_path+'val_dl_'+str(i+1)+'_'+str(j))
            for epoch in range(1, args.epochs + 1):
                model.train()
                train_loss = 0.
                train_error = 0.
                train_acc=0.
                val_loss=0.
                val_error=0.
                val_acc=0.
                for batch_idx, (dataset,ids,label) in enumerate(train_loader):
                    bag_label = label.gt(0)
                    data = dataset.unsqueeze(0)
                    data = dataset.view(ids.shape[1], dataset.shape[1])
                    if args.cuda:
                        data, bag_label = data.cuda(), bag_label.cuda()
                    data, bag_label = Variable(data), Variable(bag_label)
                    # reset gradients
                    optimizer.zero_grad()
                    # calculate loss and metrics
                    loss, acc, _, _,_ = model.calculate_objective(data, bag_label)  
                    train_loss += loss.data[0]
                    train_acc += acc
                    error, _ = model.calculate_classification_error(data, bag_label)
                    train_error += error
                    # backward pass
                    loss.backward()
                    # step
                    optimizer.step() 
                # calculate loss and error for epoch
                train_loss /= len(train_loader)
                train_error /= len(train_loader)
                train_acc /= len(train_loader)
                train_loss=train_loss.cpu().numpy()[0]
                
                #start validation
                if epoch % snapStep == 0 or epoch >= args.epochs:
                    model.eval()
                    label_list=[]
                    predicted_label_list=[]
                    with torch.no_grad():
                        for batch_idx, (dataset,ids,label) in enumerate(val_loader):
                            bag_label = label.gt(0)
                            label_list.append(bag_label.squeeze(0).tolist())
                            data = dataset.unsqueeze(0)
                            data = dataset.view(ids.shape[1], dataset.shape[1])
                            if args.cuda:
                                data, bag_label = data.cuda(), bag_label.cuda()
                            data, bag_label = Variable(data), Variable(bag_label)
                            loss,acc,_,_,_ = model.calculate_objective(data, bag_label)
                            val_loss += loss.data[0]
                            val_acc += acc
                            error, _ = model.calculate_classification_error(data, bag_label)
                            val_error += error

                    val_loss /= len(val_loader)
                    val_acc /= len(val_loader)
                    val_error /= len(val_loader)
                    val_loss=val_loss.cpu().numpy()[0]
                    if max_acc <= val_acc:
                        max_acc = val_acc
                        model_save_path = os.path.join(output+'host_'+str(i+1)+'_model.'+str(j+1))
                        torch.save(model.state_dict(), model_save_path)
                    model.train()
                    
                    csvfile=output+'virus_model_CV_5fold_new.csv'
                    results= {'Hostname': hostname,'train_Loss':train_loss,
                            'train_error':train_error,'train_acc':train_acc,'Val_Loss':val_loss,'Val_error':val_error,
                            'Val_Accuracy':val_acc} 
                    results2CSV(results, hostname,csvfile)
        