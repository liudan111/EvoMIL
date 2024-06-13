# Prediction of virus-host associations using protein language models and multiple instance learning
Dan Liu, Fran Young, David L Robertson*, Ke Yuan*
### Abstract:
Predicting virus-host associations is essential to determine the specific host species viruses interact with, and discover if new viruses infect humans and animals. Currently, the host of the majority of viruses is unknown, particularly in microbiomes. Here, we introduce EvoMIL, a deep learning method that predicts virus-host association at the species level from viral sequence only. The method combines a pre-trained large protein language model (ESM)[1] and attention-based multiple instance learning[2] to allow protein-orientated predictions. Our results show that protein embeddings capture stronger predictive signals than sequence composition features, including amino acids, DNA k-mers, and physiochemical properties. EvoMIL binary classifiers achieve AUC values of over 0.95 for all prokaryotic and roughly 0.8 to 0.9 for eukaryotic hosts. In multi-host prediction tasks, EvoMIL achieved median performance improvements in prokaryotic hosts and eukaryotic hosts. Furthermore, EvoMIL estimates the importance of single proteins in the prediction task and maps them to an embedding landscape of all viral proteins, where proteins with similar functions are distinctly clustered together.

## EvoMIL
###  A diagrammatic representation of the EvoMIL method.
First, protein sequences of viruses and virus-host interactions can be collected from the VHDB (https://www.genome.jp/virushostdb/). For each host, we get the same number of positive viruses and negative viruses, and then representative vectors of viral protein sequences are obtained by the pre-trained transformer model ESM-1b https://github.com/facebookresearch/esm. There is a host label for a set of protein sequences of each virus, attention-based MIL is applied to predict the host label for each virus, and calculate instance weights for each protein of viruses. Finally, probability between the virus and the given host is obtained to check if a virus is associated with the host.

![flowchart](https://github.com/liudan111/EvoMIL/blob/main/Figures/flowchart.pdf)

### Requirements:
    Python 3.7.7
    PyTorch ==1.7.1 (with CUDA)
    torchvision == 0.8.2
    sklearn==0.23.2
    numpy==1.21.5

### Processing steps of EvoMIL
1.  Collect viruses which are linked with hosts from VHDB https://www.genome.jp/virushostdb/, and construct the balance datasets for each host.

2.  Pre-trained transformer model ESM-1b (https://github.com/facebookresearch/esm) to calculate embeddings of each protein of viruses.

python extract.py esm1b_t33_650M_UR50S protein.fasta esm1b/ --repr_layers 33 --include mean per_tok --truncate

3.  Train binary and multi-class classification models by attention-based MIL.


### Data
1.  Data/virushostdb_update.csv
    This table includes all associations between viruses and hosts from VHDB datasets.

2.  Data/examples/pro_5fold_cv
    There is a Prokaryotic host example, with training and validation datasets generated by 5-folod cross validation, and a test dataset.

3.  Data/examples/euk_5fold_cv 
    There is an Eukaryotic host example, with training and validation datasets generated by 5-folod cross validation, and a test dataset.

4.  hostname_count.csv from Data/examples/pro_5fold_mc and Data/examples/euk_5fold_mc
    Tables respectively represented 22 Prokaryotic and 36 Eukaryotic hosts on multi-class classfication task.
     
### Codes for training and testing on  Binary classification dataset.
1. Models/model_esm1b.py
   Attention-based MIL model for binary classification.

2. Models/train.py
   Train the model by 5-fold cross validation.

3. Models/test.py
    Test 5 trained models on the test data, and obtain AUC, Accuracy and F1 score to evaluate binary models.
    
4. mil_pytorch are mil package from https://github.com/jakubmonhart/mil_pytorch.git

### Codes for training and testing on Multi-class classification dataset.
1. Models/attention_mc.py. Attention-based MIL model for multi-class classification.

2. Models/train_mc.py. Train the model by 5-fold cross validation.

3. Models/test_mc.py. Test 5 trained models on the test multi-class data, and obtain AUC, Accuracy and F1 score to evaluate multi-class models.

### Trained models on prokaryotic hosts
1.  Results/pro_mc
    We trained multi-class model on 22 prokaryotic hosts, and the trained model(host_model) can be used to test if a virus associated with these hosts.

2.  Results/pro_binary  
    Results of binary classification model Bacillus cereus from Data/examples/pro_5fold_cv. Based on 5-fold cross validation, we trained our model on prokaryotic hosts separately, then test five models on test datasets to obtain evaluation indices for each host. 


### Trained models on eukaryotic hosts
1. Results/euk_mc
   We trained multi-class model on 36 eukaryotic hosts, and it can be used for testing if a virus is associated with these hosts.

3. Results/euk_binary
    Results of binary classification model Mus musculus from Data/examples/euk_5fold_cv. Based on 5-fold cross validation, we trained our model on each eukaryotic host separately, then test five models on test datasets to obtain evaluation indices for each host. 
    
## Reference
[1] Rives A, Meier J, Sercu T, Goyal S, Lin Z, Liu J, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proc Natl Acad Sci U S A. 2021;118(15)

[2] Ilse M, Tomczak J, Welling M. Attention-based deep multiple instance learning. In: International conference on machine learning. PMLR; 2018. p. 2127–2136.
