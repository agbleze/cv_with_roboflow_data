#%%
#from coco2kitti import coco2kitti
from functools import lru_cache, wraps
from time import time
import os
from glob import glob
from pycocotools.coco import COCO

# %%
@lru_cache(maxsize=100)
def timing(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        ts = time()
        result = func(*args, **kwargs)
        te = time()
        print(f"Function '{func.__name__}' took {te-ts} seconds to run")
        return result
    return wrap


#%%
annfile_train = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/train/_annotations.coco.json"

#%%
train_folder = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/train"

train_imgs = glob(f"{train_folder}/*.jpg")

coco = COCO(annfile_train)

#%% display coc categories and supercateories
cats = coco.loadCats(coco.getCatIds())

# get names of categories
nms = [cat['name'] for cat in cats]

#%%
catNms = [cat['name'] for cat in cats]
catIds = coco.getCatIds(catNms=catNms)


#%%

for img in coco.imgs:
    print(img)
     
#%%
annIds = coco.getAnnIds(imgIds=[2240], catIds=catIds)

#%%
coco.loadAnns(annIds)



#%%

# These settings assume this script is in the annotations directory
dataDir = '..'
dataType = 'train2014'
#annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)

annFile = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/train/_annotations.coco.json"

# If this list is populated then label files will only be produced
# for images containing the listed classes and only the listed classes
# will be in the label file
# EXAMPLE:
#catNms = ['person', 'dog', 'skateboard']
catNms = nms #[]

# Check if a labels file exists and, if not, make one
# If it exists already, exit to avoid overwriting
if os.path.isdir('./labels'):
    print('Labels folder already exists - exiting to prevent badness')
else:
    os.mkdir('./labels')
    coco2kitti(catNms=nms, annFile=annFile)


