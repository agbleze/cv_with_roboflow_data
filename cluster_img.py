
#%%
from clustimage import Clustimage



# %%

cl = Clustimage(method="pca")

#%%
train_img = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/train"

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

from glob import glob

valid_dir = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/valid"

#%%
valid_imgs_list = glob(f"{valid_dir}/*.jpg")

train_imgs_list = glob(f"{train_img}/*.jpg")

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
import pandas as pd

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
from sklearn.model_selection import train_test_split


#%%

train_df, test_df = train_test_split(results_cluster_df, train_size=0.7, shuffle=True, random_state=2023,
                                    stratify=results_cluster_df[["cluster"]]
                                    )



#%%
train_df['cluster'].value_counts()

#%%
test_df['cluster'].value_counts()

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
