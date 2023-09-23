
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
        """A class with methods that determines if an image should be used a background image.
            The aim is to filter-out images that contain objects that will be used for copy-paste 
            augmentation.
        
        Args:
            coco_annotation_file (json): json file containing coco annotation
            img_class_name (_type_): name of image category that background image should not contain - using the
                                    object being used for copy-paste.
        """
        self.coco_annotation_file = coco_annotation_file
        self.img_class_name = img_class_name
        self.coco = COCO(self.coco_annotation_file)
        
        
    def check_image_contains_class(self,coco_annotation_file, image_name, class_name):
        self.image_name = image_name
        ## get the names of all images which contain this class category
        
        catIds  = coco.getCatIds(catNms=class_name)
        imgIds = coco.getImgIds(catIds=catIds)
        imgIds = coco.getImgIds(imgIds=imgIds)
        self.img_loads = coco.loadImgs(imgIds)
        self.imgs_names_for_class = [load['file_name'] for load in self.img_loads]
    
    def get_anns(self):
        self.coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)    
        ## check if image contains this class
    def _is_class_in_image(self):
        img_contains_class = [True if self.image_name in self.imgs_names_for_class else False]
        
        if not img_contains_class:
            background_image = self.image_name
        else:
            print(f'{self.image_name} have {self.img_class_name} hence not considered background image for copy paste')
            

    def plot_image_with_annotation(self, image_folder, img_name):
        
        imgs = plt.imread(f"{image_folder}/{self.imgs_names_for_class}")
        plt.imshow(imgs)
    
    
    
    def show_image(image):
        plt.figure(figsize=(15, 15))
    
    
    
    
        
