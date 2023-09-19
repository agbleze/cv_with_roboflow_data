
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

class BackgroundImageDetector(object):
    def __init__(self, coco_annotation_file, img_class_name):
        self.coco_annotation_file = coco_annotation_file
        self.img_class_name = img_class_name
        
        
    def check_image_contains_class(coco_annotation_file, image_name, class_name):
        ## get the names of all images which contain this class category
        coco = COCO(annfile_train)
        catIds  = coco.getCatIds(catNms=class_name)
        imgIds = coco.getImgIds(catIds=catIds)
        imgIds = coco.getImgIds(imgIds=imgIds)
        img_loads = coco.loadImgs(imgIds)
        imgs_names_for_class = [load['file_name'] for load in img_loads]
        
        ## check if image contains this class
        img_contains_class = [True if image_name in imgs_names_for_class else False]
        
        if not img_contains_class:
            background_image = image_name
        else:
            print(f'{image_name} have {class_name} hence not considered background image for copy paste')
            
            
    
    
    




