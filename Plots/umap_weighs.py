from time import time
import numpy as np
import pandas as pd
# For plotting
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import plotly.graph_objects as go
import matplotlib as mpl
# sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
#For standardising the dat
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np; np.random.seed(1)
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
#PCA
#Ignore warnings
import warnings
warnings.filterwarnings('ignore')
import umap
import umap.plot
import csv
import scipy
import os  
import ast
import math
class Dataload:
    def __init__(self, fileindex=0):
        self.fileindex = fileindex
        
    def create_bags(self):
      ########---test---########
            input_path = "esm1b_outputs/Prokaryote/new1/cutoff/umap/" #esm1b_outputs/Prokaryote/new/umap/10fold/,
            output_path ="esm1b_outputs/Prokaryote/new1/cutoff/umap/"
        
            pos_data = pd.read_csv(input_path + 'data_pos_weights_cutoff_top100_'+str(self.fileindex+1) + '.csv', header=None)
            x1=pos_data.iloc[:,0:1280].values
            y1=pos_data[1281].values
            print(len(x1.tolist()))
            reducer1 = umap.UMAP(n_neighbors=math.ceil(len(x1.tolist())/5),min_dist=0.1,random_state=42)
            embedding1 = reducer1.fit_transform(x1)
            print(embedding1.shape)
            
            neg_data = pd.read_csv(input_path + 'data_neg_weigths_cutoff_top100_'+str(self.fileindex+1) + '.csv', header=None)
            x2=neg_data.iloc[:,0:1280].values
            y2=neg_data[1281].values
            print(len(x2.tolist()))
            reducer2 = umap.UMAP(n_neighbors=math.ceil(len(x2.tolist())/5),min_dist=0.1,random_state=42)
            embedding2 = reducer2.fit_transform(x2)
     
            plt.rcParams["font.family"] = "Nimbus Roman"
            plt.rcParams.update({'font.size': 22})

            fig, (ax, ax2) = plt.subplots(nrows=2,figsize=(10,15))
            cmap=cm.get_cmap('viridis')
            fig.subplots_adjust(wspace=0.3)
            im  = ax.scatter(embedding1[:, 0], embedding1[:, 1], s=100, c=y1,edgecolors='none', cmap=cmap)
            im2 = ax2.scatter(embedding2[:, 0], embedding2[:, 1], s=100, c=y2,edgecolors='none', cmap=cmap) 
            ax.set_title(hostname,fontsize=30)
            ax.set_ylabel("Protein weights of positive viruses",fontsize=25)
            ax2.set_ylabel("Proteins weights of negative virsues",fontsize=25)
            fig.colorbar(im, ax=[ax,ax2],shrink=0.9)
            fig.savefig(output_path+'Prokaryote_cutoff_top100_'+str(self.fileindex+1)+'.png',dpi=300,format='png') 
            
if __name__ == "__main__":
    #VHDB:
    input_path = '/home1/2656169l/data/Prokaryote/new1/'
    virus_host_30 = pd.read_csv( input_path +'final_specise_sort_30.csv')
    length=virus_host_30.shape[0]
    # sns.set(style='white', context='notebook', rc={'figure.figsize':(8,6)})
    for i in range(length):
        load=Dataload(i)
        hostname=virus_host_30.iloc[i,0]
        load.create_bags()
        print(i)
  