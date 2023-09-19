
#%%
import os
from pycocotools.coco import COCO
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob



annfile_train = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/train/_annotations.coco.json"

coco = COCO(annfile_train)

#%%

coco.anns

#%%
coco.cats

#%%
coco.dataset

#%%
coco.imgs

#%%
coco.imgToAnns

#%% display coc categories and supercateories
cats = coco.loadCats(coco.getCatIds())

# get names of categories
nms = [cat['name'] for cat in cats]

# get supercateories names
supercat = set([cat['supercategory'] for cat in cats])

#%% get all images cntaining given category
catIds  = coco.getCatIds(catNms='leafminer')
imgIds = coco.getImgIds(catIds=catIds)

imgIds = coco.getImgIds(imgIds=imgIds)

#%%
img_loads = coco.loadImgs(imgIds)

imgs = [load['file_name'] for load in img_loads]
#imgIds = coco.getImgIds()

#%%

train_imgfiles = glob("/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/train/*.jpg")

#%%
train_imgfiles[0]

#%%

train_img = plt.imread(train_imgfiles[0])

plt.imshow(train_img)

#%%

coco
#%%
def check_image_contains_class(annotation, image_name, class_name):
    pass




