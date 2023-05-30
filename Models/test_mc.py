from __future__ import print_function
import numpy as np
import argparse
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
from attention_mc import Attention
from sklearn.metrics import roc_curve, auc
from evaluation_mc import topks_correct,topk_accuracies,results2CSV_test
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
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

def process(filename):
        input_data="../Data/examples/" + filename + "_5fold_mc/"
        output_path= '../Results/' + filename + '_mc/'
        host = pd.read_csv(input_data+'hostname_count.csv') 
        classes= host.shape[0]
        print('Init Model')
        if args.model=='attention':
            model = Attention(classes)
        if args.cuda:
            model.cuda()
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")

        test_loader=torch.load(input_data+'test_loader')
        for j in range(5):
                model.load_state_dict(torch.load(output_path + 'host_model.'+str(j+1)))
                gts, y_hat, probs = [], [], []
                with torch.no_grad():
                    test_loss=0.
                    test_acc=0.
                    #start validation
                    label_list=[]
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
                            preds,_ ,_= model(data)
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

                test_loss /= len(test_loader)
                test_acc /= len(test_loader)
                # test_error /= len(test_loader)
                
                test_loss=test_loss.cpu().numpy()
                test_acc=test_acc.cpu().numpy()

                gts, y_hat, probs = np.asarray(gts), np.asarray(y_hat), np.asarray(probs)
               
                f1 = f1_score(gts, y_hat, average='macro')
                auc_score = roc_auc_score(gts, probs, multi_class='ovr')
            
                csvfile=output_path +'virus_testset_5fold_cross_validation_epoch50_10.csv'
                results= {'AUC':str(round(np.array(auc_score).mean(),4)),'Accuracy':round(np.array(test_acc).mean(),4),'f1':round(np.array(f1).mean(),4)} 
                results2CSV_test(results,csvfile)
                
if __name__ == "__main__":
        # Eukaryota/Prokaryote
        process('pro')
        print('complete pro hosts')
        process('euk')
        print('complete euk hosts')