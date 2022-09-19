from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')
from dataloader import MnistBags
# from dataload2 import Dataload
from cv import Dataload
from model_esm1b import Attention, GatedAttention
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import pandas as pd
import os
import csv
from sklearn.metrics import cohen_kappa_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
plt.style.use('ggplot')
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                    help='number of bags in test set')
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
            fieldnames = ['Hostname','AUC','Loss','Accuracy','eer_fpr','eer_fnr','f1','spec','sens','prec']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(results) 
    else:
        with open(csvfile, 'a') as csvfile:
            print ( 'new file',csvfile)
            fieldnames = ['Hostname','AUC','Loss','Accuracy','eer_fpr','eer_fnr','f1','spec','sens','prec']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(results)        

def eer(pred, labels):
    fpr, tpr, threshold = metrics.roc_curve(labels, pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    fnr = 1 - tpr
    EER_fpr = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    EER_fnr = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    return EER_fpr, EER_fnr,auc_score

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
    f1= f1_score(pred,target,average='macro')
    # f1_val=2*(prec*sens)/(prec+sens)
    # print(f1_val)
    return acc,f1,spec,sens

if __name__ == "__main__":
    # Eukaryota/Prokaryote
    input='/home1/2656169l/data/Prokaryote/new1/'
    output='/home1/2656169l/data/Prokaryote/new1/5fold_model/'
    output_path="esm1b_outputs/Prokaryote/new1/5fold_cv/"
    virus_host_30 = pd.read_csv(input+'final_specise_sort_30.csv') #Eukaryota_final_specise_sort_30
    length=virus_host_30.shape[0]
    pd_best_acc=pd.read_csv(output+'virus_testset_AUC_5fold_best_AUC.csv')  
    markers = ["." , "," , "o" , "h" , "d" , "D", "x",'+','*','s'] #'v'
    colors = ['#1f77b4', '#FFFF00', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf','#7CFC00', '#ff7f0e', '#FFFACD', '#ADD8E6', '#F08080', '#E0FFFF', '#FF0000', '#FFA07A', '#778899', '#FFA500','#D8BFD8']
    for m in range(length):
                hostname=virus_host_30.iloc[m,0]
                test_loader=torch.load(output_path+'test_loader_'+str(m+1))
                max_index=pd_best_acc['model_CV'].tolist()
                j=max_index[m]
                model.load_state_dict(torch.load(output+'host_'+str(m+1)+'_model.'+str(j)))
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
                            bag_label = label.gt(0)
                            label_list.append(bag_label.squeeze(0).tolist())
                            data = dataset.unsqueeze(0)
                            data = dataset.view(ids.shape[1], dataset.shape[1])
                            if args.cuda:
                                data, bag_label = data.cuda(), bag_label.cuda()
                            data, bag_label = Variable(data), Variable(bag_label)
                            temp=ids
                            if(label==1):
                                for i in range(len(temp[0,])):
                                    temp[0,i]=1
                            else:
                                for i in range(len(temp[0,])):
                                    temp[0,i]=-1
                            instance_labels = temp.gt(0)
                            loss,acc,attention_weights,Y_prob,Y_hat = model.calculate_objective(data, bag_label)
                            error, predicted_label = model.calculate_classification_error(data, bag_label)
                            test_loss += loss.data[0]
                            test_acc += acc
                            test_error += error
                            Y_prob_value=Y_prob.cpu().squeeze(0).squeeze(0).tolist()
                            pred.append(Y_prob_value)
                            instance_label_list.append(instance_labels)
                            Y_hats.append(Y_hat.item())
                            predicted_label_list.append(int(predicted_label.cpu().data.numpy()[0][0]))

                test_loss /= len(test_loader)
                test_acc /= len(test_loader)
                test_error /= len(test_loader)
                test_loss=test_loss.cpu().numpy()[0]
                label_list=torch.tensor(label_list).detach()
                label_list_int=np.array(label_list, dtype='int').tolist()
                Y_hats_int=list(map(int,Y_hats))
                precision = precision_score(label_list_int,Y_hats_int,average='macro')
                acc,f1,spec,sens = get_confusion(torch.tensor(predicted_label_list), torch.tensor(label_list))
                # calculating precision and reall
                eer_fpr,eer_fnr,auc_score= eer(pred, label_list_int)
                # prec, recall, _ = precision_recall_curve(label_list_int, Y_hats_int)
                # cf_matrix =  confusion_matrix(label_list_int, Y_hats_int)
                
                plt.rcParams["font.family"] = "Nimbus Roman"
                plt.rcParams.update({'font.size': 12})
                fpr, tpr,thre = metrics.roc_curve(label_list_int,pred)
                print(thre)
                #create ROC curve
                plt.plot(fpr,tpr,alpha=.7,color=colors[m],marker=markers[8],linewidth=1.5,linestyle=':',label=hostname+" AUC="+str(round(np.array(auc_score).mean(),4)))
                plt.rc('font', size=6)    
                plt.rc('legend', fontsize=6) 
                plt.title('Roc Curve')
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.legend(loc=4)
                plt.savefig(output+'all_host_roc_new.pdf',dpi=300,format='pdf')

                # lw=2
                # fpr, tpr,thre = metrics.roc_curve(label_list_int,pred)
                # print(thre)
                # #create ROC curve
                # plt.plot(fpr,tpr,alpha=.6,color=colors[m-2],marker=markers[m-2],linewidth=1,linestyle=':',label=hostname+" AUC="+str(round(np.array(auc_score).mean(),4)))
                # plt.title('Roc Curve');
                # plt.ylabel('True Positive Rate')
                # plt.xlabel('False Positive Rate')
                # plt.legend(loc=4)
            
             
                # plt.plot(prec, recall)
                # # PrecisionRecallDisplay.from_predictions(label_list_int, Y_hats_int)
                # #add axis labels to plot
                # plt.title('Precision-Recall curve: PR={0:0.2f}'.format(precision))
                # plt.ylabel('Precision')
                # plt.xlabel('Recall')
                # plt.savefig(output+'all_host_'+str(m+1)+'.pdf',dpi=300,format='pdf')


                # group_names = ['True Neg','False Pos','False Neg','True Pos']
                # group_counts = ["{0:0.0f}".format(value) for value in
                # cf_matrix.flatten()]
                # group_percentages = ["{0:.2%}".format(value) for value in
                #      cf_matrix.flatten()/np.sum(cf_matrix)]
                # labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                # zip(group_names,group_counts,group_percentages)]
                # labels = np.asarray(labels).reshape(2,2)
                # ax=sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
                # ax.set_title('Seaborn Confusion Matrix with labels');
                # ax.set_xlabel('Predicted Values')
                # ax.set_ylabel('Actual Values');
                # ## Ticket labels - List must be in alphabetical order
                # ax.xaxis.set_ticklabels(['False','True'])
                # ax.yaxis.set_ticklabels(['False','True'])     
             
           
                # pos_index = np.where(label_list_int)[0]
                # neg_index=[i for i, val in enumerate(label_list_int) if not val]
                # pos_pred=[]
                # for index in range(len(pos_index)):
                #     pos_pred.append(pred[pos_index[index]])
                # # neg pred
                # neg_pred=[]
                # for index in range(len(neg_index)):
                #     neg_pred.append(pred[neg_index[index]])
                # # seaborn
                # # pos hist
                # sns.distplot(pos_pred, bins = 20, kde = False, hist_kws = {'color':'steelblue'},
                #             label = ('pos','hist'),norm_hist=True)
                # # neg hist
                # sns.distplot(neg_pred, bins = 20, kde = False, hist_kws = {'color':'purple'},
                #             label = ('neg','hist'),norm_hist=True)
                # # ked for pos samples
                # sns.distplot(pos_pred, hist = False, kde_kws = {'color':'red', 'linestyle':'-'},
                #             norm_hist = True, label = ('pos','ked'))
                # # ked for neg samples
                # sns.distplot(neg_pred, hist = False, kde_kws = {'color':'black', 'linestyle':'--'},
                #             norm_hist = True, label = ('neg','ked'))
                # plt.title('Distribution of prediction scores of viruses')
                # plt.xlabel('prediction score')
                # plt.ylabel('density')
                # plt.legend()
                # plt.savefig(output+'all_host_'+str(m+1)+'.pdf',dpi=300,format='pdf')