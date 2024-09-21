

#%%
import json
import cv2
import albumentations as A
import inspect
from pycocotools.coco import COCO


#%%
albu_methods = [method for method in dir(A) if callable(getattr(A, method)) and not method.startswith("__")]


# %%
augconfig_path = "/home/lin/codebase/cv_with_roboflow_data/config_auger.json"
with open(augconfig_path, "r") as f:
    augconfig = json.load(f)
    
#%%
def is_valid_albumentation_augtype(augconfig):
    augtypes = augconfig.keys()
    invalid_augtype = [augtype for augtype in augtypes if augtype not in albu_methods]  
    if invalid_augtype:
        raise ValueError(f"""Augementation Types: {invalid_augtype} are not supported by Albumentations.
                         Check the names of the augmentation provided to ensure the match a valid Albumentation method
                         """
                        )
    else:
        return True

def is_valid_albumentation_parameter(augconfig):
    augtypes = augconfig.keys()
    valid_augtype_status = is_valid_albumentation_augtype(augconfig=augconfig)
    if valid_augtype_status:
        error_messages = []
        for augtype in augtypes:
            aug_params = augconfig[augtype]
            invalid_params = [param for param in aug_params 
                              if param not in inspect.signature(getattr(A, augtype)).parameters
                              ]
            if invalid_params:
                param_error = ",".join(invalid_params) + f" is(are) not valid parameter(s) for {augtype}"
                error_messages.append(param_error)
                
        if error_messages:
            message_to_show = "\n".join(error_messages)
            raise ValueError(message_to_show)
        else:
            return True
            
def compose_albumentation_pipeline(augconfig):
    is_valid_albumentation_parameter(augconfig=augconfig)
    pipeline = []
    for augtype, aug_params in augconfig.items():
        aug_func = getattr(A, augtype)
        pipeline.append(aug_func(**aug_params))
    return A.Compose(pipeline, 
                     bbox_params=A.BboxParams(format="coco", 
                                              label_fields=["class_labels", "class_categories"]
                                              )
                     )

#%%
coco_path = "/home/lin/codebase/cv_with_roboflow_data/coco_annotation_coco.json"

#%%
coco = COCO(annotation_file=coco_path)

#%%

coco.anns
coco.loadImgs(1)

#%%
len(coco.imgToAnns)

coco.imgToAnns[1]

#%%
coco.imgs[1]["file_name"]

#%%

coco.loadAnns(1)
#%%

def get_image_and_annotations(coco, img_id):
    img = coco.imgs[img_id]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    return img, anns

#%%

img, anns = get_image_and_annotations(coco=coco, img_id=1)
img_name = img["file_name"]
img_path = f"/home/lin/codebase/cv_with_roboflow_data/subset_extract_folder/tomato_fruit/{img_name}"
img = cv2.imread(img_path)
#for ann in anns:
segms = [coco.annToMask(ann) for ann in anns]
bbox = [ann["bbox"] for ann in anns]
cat = [ann["category_id"] for ann in anns]


#%%
for ann in anns:
    coco.annToMask(ann["segmentation"])
    break


#%%
albu_compose = compose_albumentation_pipeline(augconfig=augconfig)


#%%
transformed = albu_compose(image=img, bboxes=bbox, 
             masks=segms, class_labels=cat, 
             class_categories=["ripe", "ripe", "ripe"]
             )

#%%

transformed["image"] 

#%%
transformed["bboxes"] 

#%%
transformed_masks = transformed["masks"] 

#%%
mask1 = transformed_masks[0]

#%%
from pycocotools import mask as maskUtils

#%%

maskUtils.toMask(mask1).flatten().tolist()

#%%

import numpy as np
from pycocotools import mask as maskUtils

# Example binary mask (2D numpy array)
binary_mask = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=np.uint8)

# Convert binary mask to RLE
rle = maskUtils.encode(np.asfortranarray(mask1))

# Convert RLE to polygons
polygons = maskUtils.frPyObjects(rle, mask1.shape[0], binary_mask.shape[1])

# Merge the polygons
merged_polygons = maskUtils.merge(polygons)

# Convert to list format
flattened_polygons = merged_polygons.flatten().tolist()

print(flattened_polygons)

#%%

coco.annToRLE(mask1)
#%%
transformed["class_labels"]

#%%

from PIL import Image


Image.fromarray(transformed["image"])#.show()

#%%

cv2.imshow(img)
#%%

is_valid_albumentation_parameter(augconfig=augconfig)      

#%%
inspect.signature(getattr(A,albu_methods[0])).parameters


# %% TODO: check augmented mask and bbox

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image
import torchvision.transforms.functional as F
import torch
import matplotlib.pyplot as plt

#%%
image = read_image(img_path)

bbox_converted = [[bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]] for bb in bbox]
# %%
bbox_converted = torch.tensor(bbox_converted, dtype=torch.float)

#%%
images_with_bboxes = draw_bounding_boxes(image=image, boxes=bbox_converted,
                                         colors="red", width=2
                                         )

# %%
plt.imshow(F.to_pil_image(images_with_bboxes))
plt.axis("off")
plt.show()
# %%
segms_tensor = torch.tensor(segms, dtype=torch.bool)
# %%
image_with_mask = draw_segmentation_masks(image=images_with_bboxes, masks=segms_tensor, alpha=0.9)
#%%
plt.imshow(F.to_pil_image(image_with_mask))
plt.axis("off")
plt.show()

# %% create function to take aug, visualize default img and augmentd img with mask and bbox

def augment_and_visualize(augconfig, img_path):
    
