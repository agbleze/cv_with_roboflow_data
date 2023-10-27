
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





# %%
