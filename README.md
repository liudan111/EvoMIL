# Predict virus and host interactions based on pre-trained transformer model and attention-based multiple instances learning
Dan Liu, Fran Young, Ke Yuan*, David L Robertson*
### Abstract:
Detecting virus-host interactions is essential for us to understand interaction mechanisms between virus and host and explore clues regarding diseases caused by viruses. However, host information of the majority of viruses are unknown. The virus mimics the host's patterns for escaping the immune response, viruses might contain features that are related with virus-host interactions. We introduce PreTLM-MIL, extracting protein features by a pre-trained transformer model, then applying attention-based multiple instance learning (MIL) to predict hosts and calculate weight of each protein, which can be used to explain key proteins associated with hosts. 

## PreTLM-MIL

###  The flowchart of PreTLM MIL for prediction
The flow chart of PreTLM-MIL is shown in Fig 1. First, protein sequences of viruses and virus-host interactions can be collected from the VHDB https://www.genome.jp/virushostdb/. For each host, we get the same number of positive viruses and negative viruses, and then representative vectors of viral protein sequences are obtained by the pre-trained transformer model ESM-1b https://github.com/facebookresearch/esm. There is a host label for a set of protein sequences of each virus, attention-based MIL is applied to predict the host label for each virus, and calculate instance weights for each protein of viruses. Finally, probability between the virus and the given host is obtained to check if a virus is associated with the host.

![flowchart](https://user-images.githubusercontent.com/6703505/191104200-99f5d421-4a96-4201-ae68-2bee49b060d2.png)


### Requirements:
Python3
PyTorch ==1.7.1 (with CUDA)
torchvision == 0.8.2
sklearn==0.23.2
numpy==1.21.5

### Data

### codes for training and testing on dataset.
Model/cv_save.py
Model/model_esm1b.py
Model/test.py
Model/train.py
### codes for plotting.

### trained models

Trained model_euk/ 



Trained model_pro/ 