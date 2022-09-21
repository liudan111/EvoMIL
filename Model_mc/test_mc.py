from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')
from dataloader import MnistBags
# from dataload2 import Dataload
from cv import Dataload
from attention_mc import Attention
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import pandas as pd
import os
import csv
from sklearn.metrics import cohen_kappa_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from evaluation_mc import get_confusion,topk_accuracies,results2CSV_test

# Training settings
parser = argparse.ArgumentParser(description='PyTorch PreTLM-MIL')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
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
        input='/home1/2656169l/data/Prokaryote/new1/'
        output='./Results/'
        input_data_path="./Data/example/5fold_cv_mc/"
        test_loader=torch.load(input_data_path+'test_loader')
        for j in range(5):
                model.load_state_dict(torch.load(output+'host_model.'+str(j+1)))
                gts, y_hat, probs = [], [], []
                with torch.no_grad():
                    test_loss=0.
                    test_error=0.
                    test_acc=0.
                    #start validation
                    instance_label_list=[]
                    label_list=[]
                    predicted_label_list=[]
                    pred=[]
                    Y_hats=[]
                    for batch_idx, (dataset,ids,label) in enumerate(test_loader):
                            bag_label = label
                            label_list.append(bag_label.squeeze(0).tolist())
                            data = dataset.unsqueeze(0)
                            data = dataset.view(ids.shape[1], dataset.shape[1])
                            # virus_id = ids.cpu().data.numpy()[0][0]
                            # virusname= virus_protein_data_pd.iloc[virus_id - 1,0]
                            # protein_id_list=virus_protein_data_pd.iloc[virus_id - 1,1]
                            if args.cuda:
                                data, bag_label = data.cuda(), bag_label.cuda()
                            data, bag_label = Variable(data), Variable(bag_label)
                            preds,_ = model(data)
                            loss = criterion(preds, bag_label)
                            acc = topk_accuracies(preds, bag_label, [1])[0]
                            prob_preds = F.softmax(preds, dim=1)
                            pred_label = torch.argmax(prob_preds, dim=1).item()
                            gt = bag_label.item()

                            gts.append(gt)
                            y_hat.append(pred_label)
                            probs.append(prob_preds[0].cpu().detach().numpy())                            
                            test_loss += loss
                            test_acc += acc
                            # test_error += error
                            # Y_prob_value=Y_prob.cpu().squeeze(0).squeeze(0).tolist()
                            # pred.append(Y_prob_value)
                            # instance_label_list.append(instance_labels)
                            # Y_hats.append(Y_hat.item())
                            # predicted_label_list.append(int(predicted_label.cpu().data.numpy()[0][0]))
                test_loss /= len(test_loader)
                test_acc /= len(test_loader)
                # test_error /= len(test_loader)
                
                test_loss=test_loss.cpu().numpy()
                test_acc=test_acc.cpu().numpy()
                # print(test_loss)
                gts, y_hat, probs = np.asarray(gts), np.asarray(y_hat), np.asarray(probs)
                # print(gts)
                # print(y_hat)
                # print(probs)
                # kappa_score = cohen_kappa_score(gts, y_hat)
                # cm = confusion_matrix(gts, y_hat)
                spec,sens = get_confusion(y_hat, gts)
                #calculating precision and reall
                precision = precision_score(gts, y_hat, average='macro')
                recall = recall_score(gts, y_hat, average='macro')
                f1 = f1_score(gts, y_hat, average='macro')
                auc_score = roc_auc_score(gts, probs, multi_class='ovr')
            
                csvfile=output+'virus_testset_AUC_5fold_cross_validation.csv'
                results= {'AUC':str(round(np.array(auc_score).mean(),4)),
                'Loss':round(np.array(test_loss).mean(),4),'Accuracy':round(np.array(test_acc).mean(),4),'f1':round(np.array(f1).mean(),4),'spec':round(np.array(spec).mean(),4), 
                'sens':round(np.array(sens).mean(),4),'prec':round(np.array(precision).mean(),4)} 
                results2CSV_test(results,csvfile)