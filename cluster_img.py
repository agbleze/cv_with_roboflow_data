
#%%
from clustimage import Clustimage
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import lru_cache
from typing import Union, List, NamedTuple
import abc
import cv2
from zipfile import ZipFile
from glob import glob
import shutil
from sklearn.manifold import TSNE

#%%
import plotly.express as px

df = px.data.iris()

features = df.loc[:, :'petal_width']

tsne = TSNE(n_components=3, random_state=0)
projections = tsne.fit_transform(features, )

fig = px.scatter_3d(
    projections, x=0, y=1, z=2,
    color=df.species, labels={'color': 'species'}
)
fig.update_traces(marker_size=8)
fig.show()

# %%

cl = Clustimage(method="pca")

#%%
train_img = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/train"
valid_dir = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/valid"

subset_valid_dir = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/subset_extract_folder/valid_subset"
#%%
#valid_imgs_list = glob(f"{valid_dir}/*.jpg")[:20]

#train_imgs_list = glob(f"{train_img}/*.jpg")



#%%
#for img_path in valid_imgs_list:
#    shutil.copy(img_path, 'valid_subset')

#%%
#shutil.make_archive('valid_subset', format='zip')


#%%
#len(valid_imgs_list)

#%%

valid_transformed = cl.fit_transform(subset_valid_dir)

valid_result_feat = valid_transformed['feat']

#%%
tsne_mod = TSNE(n_components=3, random_state=2023)

#%%

tsne_3feat = tsne_mod.fit_transform(valid_result_feat)

#%%
valid_transformed['xyzcoord'] = tsne_3feat
xyzcoord_df = pd.DataFrame(valid_transformed['xyzcoord'])

#%%
xyzcoord_df.rename(columns={0: "feat_0", 1: "feat_1", 2: "feat_2"})
#%%
valid_transformed.keys()
#%%
results_selected = {key: value for key, value in valid_transformed.items() if key not in ['img', 'feat', 'xycoord']}
results_selected


#%%

results_cluster_df = pd.DataFrame.from_dict(results_selected).rename(columns={'labels': 'cluster'})
results_cluster_df


#%%
fig = px.scatter_3d(
    projections, x=0, y=1, z=2,
    color=df.species, labels={'color': 'species'}
)
fig.update_traces(marker_size=8)
fig.show()



#%%
pd.concat([results_cluster_df, xyzcoord_df])
#%%
pd.DataFrame(valid_transformed)



#%%
subset_valid_raw = cl.import_data(subset_valid_dir)

#%%
subset_valid_extracted_feat = cl.extract_feat(subset_valid_raw)


#%%
tsne_mod = TSNE(n_components=3, random_state=2023)

#%%

tsne_3feat = tsne_mod.fit_transform(subset_valid_extracted_feat)

#%%


#%%
subset_valid_embed = cl.embedding(subset_valid_extracted_feat)

#%%
cl.cluster()

#%%
results = cl.results
#%%

pd.DataFrame(subset_valid_embed)

#%%
pd.DataFrame(results['feat'])


#TODO:
# Imherit the ClusterImage class
# write function to do tsne with 3_components using feat of results produced from fit_transform method of ClustImage
# writes results fo tsne(n_componets=3) to results var of ClusterImage as xyzcoord



#%%
valid_transformed
# %%
#cl.scatter(zoom=None)

#%%
#cl.scatter(zoom=1)


#%%
fig_img_cluster = cl.scatter(zoom=1, plt_all=True, figsize=(150,100))


#%%
cl.plot_unique(img_mean=False)
#%%
train_raw = cl.import_data(train_img)


#%% extract features

train_feat = cl.extract_feat(train_raw)

#%% embedding using tSNE
xycoord = cl.embedding(train_feat)

#%%
labels = cl.cluster()

#%%
results = cl.results

#%% plots
cl.clusteval.plot()

#%%
cl.scatter()

#%%
cl.plot_unique()

#%%
cl.plot()

#%%
cl.dendrogram()

#%%



#%%
result_find = cl.find(valid_imgs_list[10], k=0, alpha=0.05)

#%%
cl.plot_find()

# %%
train_result_find = cl.find(train_imgs_list[100], k=0, alpha=0.05)
#%%
cl.plot_find()
# %%

cl.fit_transform(valid_dir)

# %%
cl.scatter(zoom=None)

#%%
cl.scatter(zoom=1)


#%%
cl.scatter(zoom=1, plt_all=True, figsize=(150,100))


#%%
cl.plot_unique(img_mean=False)

#%%
cl.scatter(dotsize=10, img_mean=False)


# %%
cl.results.keys()


#TO DO
###  

# %%


# %%
results = cl.results

#%%

results.keys()

#%%
results_selected = {key: value for key, value in results.items() if key not in ['img', 'feat', 'xycoord']}
results_selected


#%%

results_cluster_df = pd.DataFrame.from_dict(results_selected).rename(columns={'labels': 'cluster'})

#%%


#%%

train_df, test_df = train_test_split(results_cluster_df, train_size=0.7, shuffle=True, random_state=2023,
                                    stratify=results_cluster_df[["cluster"]]
                                    )



#%%
train_df['cluster'].value_counts()

#%%
test_df['cluster'].value_counts()

#%%


class ImageFolder(abc.ABC):
    def __init__(self):
        pass
    @abc.abstractmethod
    def get_img_folder(self, *args, **kwargs):
        pass

class ImageFolderGetter(ImageFolder):
    def __init__(self):
        pass
    def get_img_folder(self, list_of_contents: Union[List,None] = None,
                   list_of_names: Union[List, None] = None,
                   img_folder_path: Union[str, None] = None
                   ):
        if img_folder_path:
            self.img_folder_path
        else:
            self.list_of_contents = list_of_contents
            self.list_of_names = list_of_names
            folder_name = list_of_names[0].split(".")[0]
            self.zip_file = list_of_names[0]
            with ZipFile(self.zip_file, "r") as file:
                    extract_folder = "img_extract_folder"
                    file.extractall(extract_folder)
            self.img_folder_path = os.path.join(extract_folder, folder_name)
    
class ImageClusterCreator(object):
    # def __init__(self, list_of_contents, list_of_names) -> None:
    #     self.list_of_contents = list_of_contents
    #     self.list_of_names = list_of_names
    #     folder_name = list_of_names[0].split(".")[0]
    #     self.zip_file = None
    #     with ZipFile(list_of_names[0], "r") as file:
    #             extract_folder = "img_extract_folder"
    #             file.extractall(extract_folder)
    #     self.img_folder_path = os.path.join(extract_folder, folder_name)
    @lru_cache(maxsize=None)   
    def extract_img_features(self, img_folder_path, method="pca", 
                             
                             ): 
        self.cl = Clustimage(method=method)
        self.cl.fit_transform(img_folder_path)
        
    def plot_clustered_imgs(self, zoom=1, fig_height=150, fig_width=100,
                            plt_all=True, **kwargs):
        fig_clustered_imgs = self.cl.scatter(zoom=zoom, plt_all=plt_all, 
                        figsize=(fig_height,fig_width), **kwargs
                        )
        return fig_clustered_imgs
    @property    
    def img_cluster_result_df(self):
        results = self.cl.results
        results_selected = {key: value for key, value in results.items() 
                            if key not in ['img', 'feat', 'xycoord']
                            }
        self.results_cluster_df = pd.DataFrame.from_dict(results_selected).rename(columns={'labels': 'cluster'})
        return self.results_cluster_df
    
    def split_train_test_imgs(self):
        class DataSplitReturn(NamedTuple):
            train_df: pd.DataFrame
            test_df: pd.DataFrame
            
        results_cluster_df = self.img_cluster_result_df
        train_df, test_df = train_test_split(results_cluster_df, train_size=0.7, shuffle=True, random_state=2023,
                                    stratify=results_cluster_df[["cluster"]]
                                    )
        return DataSplitReturn(train_df=train_df, test_df=test_df)
        
    def plot_unique_imgs_per_cluster(self):
        fig_unique_img_per_cluster = self.cl.plot_unique(img_mean=False)
        return fig_unique_img_per_cluster
        
        ## create a zip file of train and test data

         

# def update_output(list_of_contents, list_of_names):
#     if list_of_contents is not None:
#         #contents_test = "valid.zip"
#         folder_name = list_of_names[0].split(".")[0]
#         with ZipFile(list_of_names[0], "r") as file:
#                 extract_folder = "img_extract_folder"
#                 file.extractall(extract_folder)
                
#         img_folder = os.path.join(extract_folder, folder_name)

#%%
# len(results['img'])

# #%%
# import numpy as np
# results['feat'][210]


# #%%

# result_dict = {f"{results['filenames'][210]}": f"{list(results['feat'][210])}"}
# result_dict

# #%%
# pd.DataFrame.from_dict(result_dict)



# #%%

# len(results['feat'].flatten())
# #%%

# len(results['filenames'])


#%%

import os
bc_folder = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/backup"

if os.path.isdir(bc_folder):
    print("yes")
else: print("no")


#%%

import zipfile

filename = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/valid.zip"
with zipfile.ZipFile(filename, "r") as zipfile:
    zipfile.extractall()


#%%
contents_test = "valid.zip"
folder_name = contents_test.split(".")[0]
with ZipFile(contents_test, "r") as file:
        extract_folder = "extract_folder"
        file.extractall(extract_folder)
        
img_folder = os.path.join(extract_folder, folder_name)
        
img = glob(f"{img_folder}/*.jpg")[0]

#%%

img

#%%

#import shutil

#shutil.make_archive("labels", "zip")

#%% 
# TO DO: First 3 are hackerton presentation critical
# bUILD a platform that allows you to upload a folder of images  / zip files
# undertakes image clustering and visualizes the results
# shows a table of images names and the cluster they belong to

## TODO: These are nice to have for fuller functionality of the platform
# Functionality for data spliting by sampling from different groups to increase dissimilarity
# Allows for downloading of data into train, test, validation, validation 
# to increase randomness and generalizability



# features to build
## upload button
## `cluster images` button -- only shows after uploading images
##--- Default determines optimal clusters to use
## Select Option to determine the number of clusters to use
## Graphs are for visualization of clustered images  -- matplotlib graph is used
##--- Option to customize graph with 
## Table visualization of cluster images names and clusters

## Nice to have
## Data split button
##--- After clicking on the split button, two buttons should be created for the download of the train and test data splitted
##--- Table of summary staticts of the splits shown -- % of images per cluster / target labels in train and test images / multiple labels for multiple object detection can be combined as a single label




# %%


if __name__ == "__main__":
    pass