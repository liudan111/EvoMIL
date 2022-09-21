from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from attention_mc import Attention
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import pandas as pd
import os
import csv
from sklearn.metrics import f1_score
from evaluation_mc import get_confusion, topks_correct,topk_accuracies,results2CSV_train

# Training settings
parser = argparse.ArgumentParser(description='PyTorch PreTLM-MIL')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
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
# elif args.model=='gated_attention':
#     model = GatedAttention()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
criterion = torch.nn.CrossEntropyLoss(reduction="mean")
   
if __name__ == "__main__":
        # Eukaryota/Prokaryote
        input='./Data/'
        output='./Results/'
        input_data_path="./Data/example/5fold_cv_mc/"
        snapStep=5 #every 5 epochs we will validate our validation set once
        for j in range(5):
            print('Start Training')
            max_acc = 0                
            train_loader=torch.load(input_data_path+'train_dl_'+str(j))
            val_loader=torch.load(input_data_path+'val_dl_'+str(j))
            # train_loader=train_loader_list[j]
            # val_loader=val_loader_list[j]
            for epoch in range(1, args.epochs + 1):
                model.train()
                train_loss = 0.
                train_error = 0.
                train_acc=0.
                val_loss=0.
                val_error=0.
                val_acc=0.
                for batch_idx, (dataset,ids,label) in enumerate(train_loader):
                    bag_label = label
                    # bag_label = label.gt(0)
                    data = dataset.unsqueeze(0)
                    data = dataset.view(ids.shape[1], dataset.shape[1])
                    if args.cuda:
                        data, bag_label = data.cuda(), bag_label.cuda()
                    data, bag_label = Variable(data), Variable(bag_label)

                    # reset gradients
                    optimizer.zero_grad()
                    # calculate loss and metrics

                    preds,_ = model(data)
                    loss_step = criterion(preds, bag_label)
                    # loss, acc, _, _,_ = model.calculate_objective(data, bag_label)  
                    # error, _ = model.calculate_classification_error(data, bag_label)
                    # train_error += error
                    # backward pass
                    loss_step.backward()
                    # step
                    optimizer.step() 
                    
                    acc = topk_accuracies(preds, bag_label, [1])[0]
                    train_loss += loss_step
                    train_acc += acc
                # calculate loss and error for epoch
                train_loss /= len(train_loader)
                # train_error /= len(train_loader)
                train_acc /= len(train_loader)
                # train_loss=train_loss.cpu().numpy()[0]
                
                #start validation
                if epoch % snapStep == 0 or epoch >= args.epochs:
                    model.eval()
                    label_list=[]
                    predicted_label_list=[]
                    with torch.no_grad():
                        for batch_idx, (dataset,ids,label) in enumerate(val_loader):
                            # bag_label = label.gt(0)
                            bag_label = label
                            label_list.append(bag_label.squeeze(0).tolist())
                            data = dataset.unsqueeze(0)
                            data = dataset.view(ids.shape[1], dataset.shape[1])
                            if args.cuda:
                                data, bag_label = data.cuda(), bag_label.cuda()
                            data, bag_label = Variable(data), Variable(bag_label)

                            preds,_ = model(data)
                            loss_step = criterion(preds, bag_label)
                            acc = topk_accuracies(preds, bag_label, [1])[0]
                            # loss,acc,_,_,_ = model.calculate_objective(data, bag_label)
                            val_loss += loss_step
                            val_acc += acc

                            # error, _ = model.calculate_classification_error(data, bag_label)
                            # val_error += error

                    val_loss /= len(val_loader)
                    val_acc /= len(val_loader)
                    # val_error /= len(val_loader)
                    # val_loss=val_loss.cpu().numpy()[0]
                    if max_acc <= val_acc:
                        max_acc = val_acc
                        model_save_path = os.path.join(output+'host_model.'+str(j+1))
                        torch.save(model.state_dict(), model_save_path)
                    model.train()
                    
                    csvfile= output+'virus_model_CV_5fold.csv'
                    results= {'train_Loss':train_loss,
                            'train_acc':train_acc,'Val_Loss':val_loss,
                            'Val_Accuracy':val_acc} 
                    results2CSV_train(results,csvfile)
        