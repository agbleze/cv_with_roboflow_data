
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


#%%

pd.DataFrame.from_dict(cl.results)
# %%
results = cl.results

#%%

results.keys()

#%%
pd.DataFrame.from_dict(results[[]]).rename(columns={'labels': })

#%%

results#[['labels', 'filenames']]

#%%
results_selected = {key: value for key, value in results.items() if key not in ['img', 'feat', 'xycoord']}
results_selected


#%%

pd.DataFrame.from_dict(results_selected)

#%%
len(results['img'])

#%%
import numpy as np
results['feat'][210]


#%%

result_dict = {f"{results['filenames'][210]}": f"{list(results['feat'][210])}"}
result_dict

#%%
pd.DataFrame.from_dict(result_dict)



#%%

len(results['feat'].flatten())
#%%

len(results['filenames'])




#%% 
# TO DO:
# bUILD a platform that allows you to upload a folder of images 
# undertakes image clustering and visualizes the results
# Functionality for data spliting by sampling from different groups to increase dissimilarity
# Allows for downloading of data into train, test, validation, validation 
# to increase randomness and generalizability





# %%
