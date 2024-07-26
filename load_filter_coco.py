
#%%
import os
from pycocotools.coco import COCO
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
from __future__ import division



annfile_train = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/train/_annotations.coco.json"

#%%
train_folder = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/train"

train_imgs = glob(f"{train_folder}/*.jpg")

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

train_img = plt.imread(train_imgfiles[70])

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
    
    

#%%    
def show(image):
    plt.figure(figsize=(15, 15))
    plt.imshow(image, interpolation="nearest")
    

def show_hsv(hsv):
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    show(rgb)    
    
def show_mask(mask):
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap="gray")
    
def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img = cv2.addWeighted(src1=rgb_mask, alpha=0.5, src2=image, beta=0.5, gamma=0)
    show(img)
    
# def find_biggest_contour(image):
#     image = image.copy()
#     im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
#     biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    
#     mask = np.zeros(image.shape, np.uint8)
#     cv2.drawContours(image=mask, contours=[biggest_contour],
#                      contourIdx=-1, color=255, 
#                      thickness=-1
#                      )
#     return biggest_contour, mask


def find_biggest_contour(image):
    image = image.copy()
    im2,contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
 
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask


def circle_contour(image, contour):
    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(img=image_with_ellipse, box=ellipse, color=(0,255,0), thinckness=2)
    return image_with_ellipse


#%% loading image display
image = cv2.imread(train_imgs[70]) 
show(image)  


#%%
image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
show(image)

#%%
cv2.COLOR_BGR2RGB

#%% resize image

max_dimension = max(image.shape)        

scale = 700/max_dimension

img_resize = cv2.resize(src=image, dsize=None, fy=scale, fx=scale)  
show(img_resize)      

# %%
#img_resize[:,8]

# %% bluring with gaussian kernel
image_blur = cv2.GaussianBlur(src=img_resize, ksize=(7,7), sigmaX=0)
show(image_blur)

#%% convert rgb to hsv
image_blur_hsv = cv2.cvtColor(src=image_blur, code=cv2.COLOR_RGB2HSV)
show(image_blur_hsv)

#%% ####### preparing mask ######
# create mask that detect specific color spectrum
# mask perform feature extraction using color
# mask 2 uses brightness factors

#%% filter by color
min_red = np.array([0, 100, 80])
max_red = np.array([180, 256, 256])
mask1 = cv2.inRange(src=image_blur_hsv, lowerb=min_red, upperb=max_red)
show(mask1)

#%% filter by brightness
min_red_brightness = np.array([0, 100, 80])
max_red_brightness = np.array([180, 256, 256])
mask2 = cv2.inRange(src=image_blur_hsv, lowerb=min_red_brightness, 
                    upperb=max_red_brightness
                    )

show(mask2)

#%% concatenate both masks for better feature extraction
mask = mask1 + mask2
show(mask)

#%% ############### post processing of mask #########
kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(15,15))
mask_closed = cv2.morphologyEx(src=mask, op=cv2.MORPH_CLOSE, kernel=kernel)
show(mask_closed)

#%%
mask_clean = cv2.morphologyEx(src=mask_closed, op=cv2.MORPH_OPEN, kernel=kernel)
show(mask_clean)

#%% ###### applying mask #####
# extract biggest bounding box
big_contour, red_mask = find_biggest_contour(mask_clean)
#show(big_contour)
#show(red_mask)

#%% apply mask
overlay = overlay_mask(red_mask, image)

# draw bounding box
circled = circle_contour(overlay, big_contour)
show(circled)





# %%
import pandas as pd
import json

f = open("/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/train/_annotations.coco.json")

annfile_json = json.load(f)

#%%
pd.json_normalize(annfile_json)
 
 # %%
 
 
 
 
