

#%%
import json
import cv2
import albumentations as A
import inspect
from pycocotools.coco import COCO


#%%
[method for method in dir(A) if callable(getattr(A, method)) and not method.startswith("__")]


# %%
