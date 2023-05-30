from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import precision_score,f1_score
from EvoMIL.Models.model import Attention,GatedAttention
import pandas as pd
from evaluation import aucscore,get_confusion,results2CSV_test

# test settings
parser = argparse.ArgumentParser(description='PyTorch PreTLM-MIL')
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

"""
We make changes on Attention_based MIL (Copyright © 2018 ilse.maximilian@gmail.com.) to test viral protein embeddings.
"""
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
        
if __name__ == "__main__":
    # Eukaryota/Prokaryote
    output='../Results/euk_binary/'
    input_data_path="../Data/examples/euk_5fold_cv/"
    # virus_pos_neg_path= input+'virus_' 
    virus_host = pd.read_csv(input_data_path+'hostname_count.csv') 
    length=virus_host.shape[0]
    for m in [4]:
        hostname=virus_host.iloc[m,0]
        test_loader=torch.load(input_data_path + hostname + '_test_loader')
        for j in range(5):
                model.load_state_dict(torch.load(output + hostname+'_model.'+str(j+1)))
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
                            # test_error += error
                            Y_prob_value=Y_prob.cpu().squeeze(0).squeeze(0).tolist()
                            pred.append(Y_prob_value)
                            instance_label_list.append(instance_labels)
                            Y_hats.append(Y_hat.item())
                            predicted_label_list.append(int(predicted_label.cpu().data.numpy()[0][0]))

                test_loss /= len(test_loader)
                test_acc /= len(test_loader)
                # test_error /= len(test_loader)
                test_loss=test_loss.cpu().numpy()[0]
                label_list=torch.tensor(label_list).detach()
                label_list_int=np.array(label_list, dtype='int').tolist()
                Y_hats_int=list(map(int,Y_hats))
                acc,spec,sens = get_confusion(torch.tensor(predicted_label_list), torch.tensor(label_list))
                # calculating precision and reall
                f1= f1_score(label_list_int,Y_hats_int)  
                prec = precision_score(label_list_int, Y_hats_int)
                # recall = recall_score(label_list_int, pred)
                auc_score= aucscore(pred, label_list_int)

                csvfile=output+'virus_testset_5fold_cross_validation_epoch150_10.csv'
                results= {'Hostname': hostname,'AUC':str(round(np.array(auc_score).mean(),4)),'Accuracy':round(np.array(acc).mean(),4),'f1':round(np.array(f1).mean(),4),'spec':round(np.array(spec).mean(),4), 'sens':round(np.array(sens).mean(),4),'prec':round(np.array(prec).mean(),4)} 
                results2CSV_test(results, hostname,csvfile)