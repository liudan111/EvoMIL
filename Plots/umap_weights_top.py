from time import time
import numpy as np
import pandas as pd
# For plotting
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import plotly.graph_objects as go
import matplotlib as mpl
import seaborn as sns
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
#For standardising the dat
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np; np.random.seed(1)
import matplotlib.font_manager
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
#PCA
#Ignore warnings
import warnings
warnings.filterwarnings('ignore')
import umap
import umap.plot
import math
import torch
class Dataload:
    def __init__(self, fileindex=0):
        self.fileindex = fileindex
        
    def create_bags(self):
      ########---test---########
            # color by protein size
            input_path = "esm1b_outputs/Prokaryote/new1/umap/5foldcv/" #esm1b_outputs/Prokaryote/new/umap/10fold/,  esm1b_outputs/Prokaryote/new1/umap/5foldcv/,esm1b_outputs/Eukaryota/new1/umap/10foldcv/
            input_new1='/home1/2656169l/data/Prokaryote/new1/'
            output_path ="esm1b_outputs/Prokaryote/new1/cutoff/umap/"
            pos_virus_type=pd.read_csv(input_new1+'virus_family_pos_'+str(self.fileindex+1)+'.csv',header=None)
            neg_virus_type=pd.read_csv(input_new1+'virus_family_neg_'+str(self.fileindex+1)+'.csv',header=None)

            pos_data = pd.read_csv(input_path + 'data_pos_weights_top5_allvirus_'+ str(self.fileindex+1) + '.csv', header=None)
            neg_data = pd.read_csv(input_path +'data_neg_weigths_top5_allvirus_'+ str(self.fileindex+1) + '.csv', header=None)

            anno_path='/home1/2656169l/data/Prokaryote/new1/'
            protein_id_annotation_pro_pd=pd.read_csv(anno_path+'pro_acc_annotation_'+str(self.fileindex+1)+'.csv')

            protein_ids_pos=pos_data.iloc[:,1282].tolist()
            anno_list_pos=[]
            for z in range(len(protein_ids_pos)):
                index = np.where(protein_id_annotation_pro_pd['protein_id_pos']==protein_ids_pos[z].split('_contigs')[0])[0].tolist()
                if(len(index)>0):
                    anno_list_pos.append(protein_id_annotation_pro_pd.iloc[index[0],2])
                else:
                    anno_list_pos.append('no annotation')

            protein_ids_neg=neg_data.iloc[:,1282].tolist()
            anno_list_neg=[]
            for z in range(len(protein_ids_neg)):
                index = np.where(protein_id_annotation_pro_pd['protein_id_pos']==protein_ids_neg[z].split('_contigs')[0])[0].tolist()
                if(len(index)>0):
                    anno_list_neg.append(protein_id_annotation_pro_pd.iloc[index[0],2])
                else:
                    anno_list_neg.append('no annotation')

            pos_data['anno_list']=anno_list_pos
            neg_data['anno_list']=anno_list_neg

            virus_family_pos=[]
            for j in range(pos_data.shape[0]):
                position=pos_data.iloc[j,1280]
                virus_family_pos.append(pos_virus_type.iloc[position-1,0]) 
                
            virus_family_neg=[]
            for j in range(neg_data.shape[0]):
                position=neg_data.iloc[j,1280]
                virus_family_neg.append(neg_virus_type.iloc[position-pos_virus_type.shape[0]-1,0])
 
            pos_data['virus_family']= virus_family_pos
            neg_data['virus_family']= virus_family_neg


            # pos_data.iloc[:,[1282,1283,1284]].to_csv(output_path+'pos_data_allinfo_'+str(self.fileindex+1)+'.csv')
            # neg_data.iloc[:,[1282,1283,1284]].to_csv(output_path+'neg_data_allinfo_'+str(self.fileindex+1)+'.csv')

            #step2 pos/neg
            x1=pos_data.iloc[:,0:1280].values
            reducer1 = umap.UMAP(n_neighbors=math.ceil(len(x1.tolist())/5),min_dist=0.1,random_state=42)
            embedding1 = reducer1.fit_transform(x1)

            x2=neg_data.iloc[:,0:1280].values
            reducer2 = umap.UMAP(n_neighbors=math.ceil(len(x2.tolist())/5),min_dist=0.1,random_state=42)
            embedding2 = reducer2.fit_transform(x2)  

# figure1
            plt.rcParams["font.family"] = "Nimbus Roman"
            plt.rcParams.update({'font.size': 22})
            fig, (ax, ax2) = plt.subplots(nrows=2,figsize=(10,15))
            cmap=cm.get_cmap('viridis')
            fig.subplots_adjust(wspace=0.3)
            im  = ax.scatter(embedding1[:, 0], embedding1[:, 1], s=30, c='brown',edgecolors='none', cmap='Spectral',alpha=0.5)

            im2 = ax2.scatter(embedding2[:, 0], embedding2[:, 1], s=30, c='purple',edgecolors='none', cmap='Spectral',alpha=0.5) 
       
            ax.set_title('UMAP projection of positive viruses on '+ hostname,fontsize=20)
            ax.set_xlabel("UMAP Dim1: protein embedding",fontsize=20)
            ax.set_ylabel("UMAP Dim2: protein embedding",fontsize=20)
            ax2.set_title('UMAP projection of negative viruses on '+ hostname,fontsize=20)
            ax2.set_xlabel("UMAP Dim1: protein embedding",fontsize=20)
            ax2.set_ylabel("UMAP Dim2: protein embedding",fontsize=20)
            ax.tick_params(axis ='both', which ='major', 
               labelsize = 16)
            plt.gca().set_aspect('equal', 'datalim')
            fig.savefig(output_path+'Prokaryote_top5_pos_neg_'+str(self.fileindex+1)+'.png',dpi=300,format='png')
            plt.close('all')

     # figure2
            # dict1 = {}
            # for key in anno_list_pos:
            #     dict1[key] = dict1.get(key, 0) + 1

            # dict2 = {}
            # for key in anno_list_neg:
            #     dict2[key] = dict2.get(key, 0) + 1


            # pd.DataFrame.from_dict(dict(sorted(dict1.items(), key=lambda item: item[1],reverse=True)), orient = 'index').to_csv(output_path+'pos_anno_count_supp_'+str(self.fileindex+1)+'.csv')
      
            # pd.DataFrame.from_dict(dict(sorted(dict2.items(), key=lambda item: item[1],reverse=True)), orient = 'index').to_csv(output_path+'neg_anno_count_supp_'+str(self.fileindex+1)+'.csv')

            # pos_ann_list_10=list(dict(sorted(dict1.items(), key=lambda item: item[1],reverse=True)).keys())[:10]
            
            # neg_ann_list_10=list(dict(sorted(dict2.items(), key=lambda item: item[1],reverse=True)).keys())[:10]

            # pos_index_list=[]
            # for i in range(len(pos_ann_list_10)):
            #     pos_index=np.where(pos_data['anno_list']==pos_ann_list_10[i])
            #     pos_index_list = pos_index_list + pos_index[0].tolist()
            # pos_data_select=pos_data.iloc[pos_index_list,:]

            # neg_index_list=[]
            # for i in range(len(neg_ann_list_10)):
            #     neg_index=np.where(neg_data['anno_list']==neg_ann_list_10[i])
            #     neg_index_list = neg_index_list + neg_index[0].tolist()
            # neg_data_select=neg_data.iloc[neg_index_list,:]



            # for the top5 proteins, we only present 10 protein functions
            # fig, (ax, ax2) = plt.subplots(nrows=2,figsize=(15,20))
            # cmap=cm.get_cmap('viridis')
            # fig.subplots_adjust(wspace=0.3)
            # dict_ann_pos_color = {}
            # index=list(range(len(pos_ann_list_10)))
            # for item in range(len(index)):
            #     dict_ann_pos_color[pos_ann_list_10[item]]=index[item]
            # dict_ann_neg_color = {}
            # index=list(range(len(neg_ann_list_10)))
            # for item in range(len(index)):
            #     dict_ann_neg_color[neg_ann_list_10[item]]=index[item]

            # pos_data_select_scatter= pd.DataFrame({'x':embedding1[:, 0].tolist(),'y':embedding1[:, 1].tolist(),'anno_list':pos_data_select['anno_list'].tolist(),'virus_family':pos_data_select['virus_family'].tolist()})
            # neg_data_select_scatter=pd.DataFrame({'x':embedding2[:, 0].tolist(),'y':embedding2[:, 1].tolist(),'anno_list':neg_data_select['anno_list'].tolist(),'virus_family':neg_data_select['virus_family'].tolist()})

            # im=ax.scatter(embedding1[:, 0], embedding1[:, 1], s=30, c=pos_data.anno_list.astype('category').cat.codes,edgecolors='none', cmap='Spectral',alpha=0.8) #c=[sns.color_palette()[x] for x in pos_data_select.anno_list.map(dict_ann_pos_color)] /c=pos_data_select.anno_list.astype('category').cat.codes /hue='anno_list',
 
            # ax.set_title('UMAP projection of protein annotation of viruses on '+ hostname,fontsize=20)
            # ax.set_xlabel("UMAP Dim1: protein embedding",fontsize=20)
            # ax.set_ylabel("UMAP Dim2: protein embedding",fontsize=20)
            # ax.tick_params(axis ='both', which ='major', 
            #    labelsize = 16)

            # im2=ax2.scatter(embedding2[:, 0], embedding2[:, 1], s=30,c=neg_data.anno_list.astype('category').cat.codes,edgecolors='none', cmap='Spectral',alpha=0.8) 
            # ax2.set_xlabel("UMAP Dim1: protein embedding",fontsize=20)
            # ax2.set_ylabel("UMAP Dim2: protein embedding",fontsize=20)
            # ax2.tick_params(axis ='both', which ='major', 
            #    labelsize = 16)

            # sp_names1=pd.unique(pos_data.anno_list.values).tolist()
            # sp_names2=pd.unique(neg_data.anno_list.values).tolist()

            # neg_data.anno_list.astype('category').cat.codes
            # ax.legend(handles=im.legend_elements()[0], 
            # labels=sp_names1,loc='best',
            # title="Protein Annotation")
            # ax2.legend(handles=im2.legend_elements()[0], 
            # labels=sp_names2,loc='best',
            # title="Protein Annotation")
            # fig.colorbar(im)
            # fig.colorbar(im2)

            # fig.savefig(output_path+'Prokaryote_annotation_top5_'+str(self.fileindex+1)+'.png',dpi=300,format='png') 
            # plt.close('all')

# # figure3
            # dict3 = {}
            # for key in virus_family_pos:
            #     dict3[key] = dict3.get(key, 0) + 1

            # dict4 = {}
            # for key in virus_family_neg:
            #     dict4[key] = dict4.get(key, 0) + 1


            # pd.DataFrame.from_dict(dict(sorted(dict3.items(), key=lambda item: item[1],reverse=True)), orient = 'index').to_csv(output_path+'pos_family_count_supp_'+str(self.fileindex+1)+'.csv')
      
            # pd.DataFrame.from_dict(dict(sorted(dict4.items(), key=lambda item: item[1],reverse=True)), orient = 'index').to_csv(output_path+'neg_family_count_supp_'+str(self.fileindex+1)+'.csv')

            # print(pd.unique(neg_data.virus_family.values))

            # print(pd.unique(neg_data.virus_family.astype('category').cat.codes))

            # print(neg_data.virus_family.astype('category').cat.codes.tolist())


            # fig, (ax, ax2) = plt.subplots(nrows=2,figsize=(10,15))
            # cmap=cm.get_cmap('viridis')
            # fig.subplots_adjust(wspace=0.3)
            # im  = ax.scatter(embedding1[:, 0], embedding1[:, 1], s=10, c=pos_data.virus_family.astype('category').cat.codes,edgecolors='none', cmap='Spectral',alpha=0.8)
            # im2 = ax2.scatter(embedding2[:, 0], embedding2[:, 1], s=10, c=neg_data.virus_family.astype('category').cat.codes,edgecolors='none', cmap='Spectral',alpha=0.8) 
            # # sp_names1=pd.unique(pos_data.virus_family.values).tolist()
            # # sp_names2=pd.unique(neg_data.virus_family.values).tolist()
            # # ax.legend(handles=im.legend_elements()[0], 
            # # labels=sp_names1,
            # # title="virus family")
            # # ax2.legend(handles=im2.legend_elements()[0], 
            # # labels=sp_names2,
            # # title="virus family")
            # ax.set_title('UMAP projection of virus family of viruses on '+ hostname,fontsize=20)
            # ax.set_xlabel("UMAP Dim1: protein embedding",fontsize=20)
            # ax.set_ylabel("UMAP Dim2: protein embedding",fontsize=20)
            # ax2.set_xlabel("UMAP Dim1: protein embedding",fontsize=20)
            # ax2.set_ylabel("UMAP Dim2: protein embedding",fontsize=20)
            # ax.tick_params(axis ='both', which ='major', 
            #    labelsize = 16)
            # ax2.tick_params(axis ='both', which ='major', 
            #    labelsize = 16)
            # fig.savefig(output_path+'Prokaryote_family_top5_nolegend_'+str(self.fileindex+1)+'.png',dpi=300,format='png') 
            # plt.close('all')
            
if __name__ == "__main__":
    #VHDB:
    input_path = '/home1/2656169l/data/Prokaryote/new1/'
    virus_host_30 = pd.read_csv( input_path +'final_specise_sort_30.csv')
    length=virus_host_30.shape[0]
    # sns.set(style='white', context='notebook', rc={'figure.figsize':(8,6)})
    # [0,1,2,3,4,5,6,12,9,7,20,11] total,,  sup [6,12,9,7,20,11]
    # char_item=97 
    for i in [0,1]:
        load=Dataload(i)
        hostname=virus_host_30.iloc[i,0]
        load.create_bags()
        # char_item=char_item+1
  