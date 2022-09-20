# Predict virus and host interactions based on pre-trained transformer model and attention-based multiple instances learning
Dan Liu, Fran Young, Ke Yuan*, David L Robertson*
### Abstract:
Detecting virus-host interactions is essential for us to understand interaction mechanisms between virus and host and explore clues regarding diseases caused by viruses. However, host information of the majority of viruses are unknown. The virus mimics the host's patterns for escaping the immune response, viruses might contain features that are related with virus-host interactions. We introduce PreTLM-MIL, extracting protein features by a pre-trained transformer model, then applying attention-based multiple instance learning (MIL) to predict hosts and calculate weight of each protein, which can be used to explain key proteins associated with hosts. 

## PreTLM-MIL
###  The flowchart of PreTLM-MIL
First, protein sequences of viruses and virus-host interactions can be collected from the VHDB https://www.genome.jp/virushostdb/. For each host, we get the same number of positive viruses and negative viruses, and then representative vectors of viral protein sequences are obtained by the pre-trained transformer model ESM-1b https://github.com/facebookresearch/esm. There is a host label for a set of protein sequences of each virus, attention-based MIL is applied to predict the host label for each virus, and calculate instance weights for each protein of viruses. Finally, probability between the virus and the given host is obtained to check if a virus is associated with the host.

![flowchart](https://user-images.githubusercontent.com/6703505/191104200-99f5d421-4a96-4201-ae68-2bee49b060d2.png)


### Requirements:
    Python3
    PyTorch ==1.7.1 (with CUDA)
    torchvision == 0.8.2
    sklearn==0.23.2
    numpy==1.21.5

### Processing steps of PreTLM-MIL
1.  From VHDB
2. Pre-trained transformer model ESM-1b to calculate emendings of protein viruses 
### Data
1. example/5fold_cv 
    This is a cross validation dataset 
2. final_specise_sort_30_pro.csv
 
3. final_specise_sort_45_euk.csv

4.  virushostdb_update.csv
### Codes for training and testing on dataset (Binary classification).
1. Model/model_esm1b.py

2. Model/test.py

3. Model/train.py

### Codes for training and testing on dataset (Multi-class classification).
1. Model_mc/attention_mc.py

2. Model_mc/test.py

3. Model_mc/train.py

### Trained models on prokaryotic hosts
1. Trained model_pro/best_model_acc 

2. Trained model_pro/best_model_auc 

3. Trained model_pro/final_model
### Trained models on eukaryotic hosts
1. Trained model_euk/best_model_acc 

2. Trained model_euk/best_model_auc 

3. Trained model_euk/final_model