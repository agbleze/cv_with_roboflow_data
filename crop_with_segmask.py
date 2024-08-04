
#%%
from pycocotools.coco import COCO
import cv2
import numpy as np
from PIL import Image
from clusteval import clusteval
import pandas as pd
import json
from feat import get_object_features
#%% Load COCO annotations
coco = COCO('/home/lin/codebase/cv_with_roboflow_data/coco_annotation_coco.json')

#%% Get image IDs
# objects_in_img = {}
# img_ids = coco.getImgIds()
# for img_id in img_ids:
#     img_info = coco.loadImgs(img_id)[0]
#     img_path = '/home/lin/codebase/cv_with_roboflow_data/tomato_fruit/' + img_info['file_name']
#     image = cv2.imread(img_path)

#     # Get annotation IDs for the image
#     ann_ids = coco.getAnnIds(imgIds=img_id)
#     anns = coco.loadAnns(ann_ids)
#     img_obj = []
#     for ann in anns:
#         segmentation = ann['segmentation']
#         mask = coco.annToMask(ann)

#         # Find contours
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             x, y, w, h = cv2.boundingRect(contour)
#             cropped_object = image[y:y+h, x:x+w]
#             img_obj.append(cropped_object)

#             # Save or display the cropped object
#             #cv2.imshow('Cropped Object', cropped_object)
#             #cv2.waitKey(0)
#             #cv2.destroyAllWindows()
#         objects_in_img[img_id] = img_obj


#%%
import os
def get_objects(imgname, coco, img_dir):
    val = [obj for obj in coco.imgs.values() if obj["file_name"] == imgname][0]
    img_id = val['id']
    print(val)
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, imgname)
    image = cv2.imread(img_path)

    # Get annotation IDs for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    img_obj = []
    for ann in anns:
        #segmentation = ann['segmentation']
        mask = coco.annToMask(ann)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cropped_object = image[y:y+h, x:x+w]
            img_obj.append(cropped_object)
    os.makedirs(name="crop_objs", exist_ok=True)        
    for img_count, each_img_obj in enumerate(img_obj):
        cv2.imwrite(filename=f"crop_objs/img_obj_{img_count}.png",img=each_img_obj)
    return img_obj


# %%
tomato_coco_path = "/home/lin/codebase/cv_with_roboflow_data/tomato_coco_annotation/annotations/instances_default.json"
img_dir = "/home/lin/codebase/cv_with_roboflow_data/images"
coco = COCO(annotation_file=tomato_coco_path)


import os
import cv2
import numpy as np

def get_objects(imgname, coco, img_dir):
    try:
        val = next(obj for obj in coco.imgs.values() if obj["file_name"] == imgname)
    except StopIteration:
        raise ValueError(f"Image {imgname} not found in COCO dataset.")
    
    img_id = val['id']
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, imgname)
    image = cv2.imread(img_path)

    # Get annotation IDs for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    img_obj = []

    for ann in anns:
        mask = coco.annToMask(ann)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cropped_object = image[y:y+h, x:x+w]

            # Apply the mask to the cropped object
            mask_cropped = mask[y:y+h, x:x+w]
            cropped_object = cv2.bitwise_and(cropped_object, cropped_object, mask=mask_cropped)
            
            # Remove the background (set to transparent)
            cropped_object = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2BGRA)
            cropped_object[:, :, 3] = mask_cropped * 255

            img_obj.append(cropped_object)
    
    os.makedirs(name="crop_objs", exist_ok=True)
    for img_count, each_img_obj in enumerate(img_obj):
        cv2.imwrite(filename=f"crop_objs/img_obj_{img_count}.png", img=each_img_obj)
    
    return img_obj


import os
import cv2
import numpy as np
from typing import List, Dict

def get_objects_keep_imgdim(imgname, coco, img_dir) -> List:
    try:
        val = next(obj for obj in coco.imgs.values() if obj["file_name"] == imgname)
    except StopIteration:
        raise ValueError(f"Image {imgname} not found in COCO dataset.")
    
    img_id = val['id']
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, imgname)
    image = cv2.imread(img_path)

    # Get annotation IDs for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    img_obj = []

    for ann in anns:
        mask = coco.annToMask(ann)

        # Apply the mask to the image to get the segmented object
        segmented_object = cv2.bitwise_and(image, image, mask=mask)
        
        # Remove the background (set to transparent)
        segmented_object = cv2.cvtColor(segmented_object, cv2.COLOR_BGR2BGRA)
        segmented_object[:, :, 3] = mask * 255

        img_obj.append(segmented_object)
    
    os.makedirs(name="crop_objs", exist_ok=True)
    for img_count, each_img_obj in enumerate(img_obj):
        cv2.imwrite(filename=f"crop_objs/img_obj_{img_count}.png", img=each_img_obj)
    
    return img_obj

#%%
objects = get_objects(imgname="0.jpg", coco=coco, img_dir=img_dir)

#%%

Image.fromarray(cv2.cvtColor(objects[0], cv2.COLOR_BGR2BGRA))
#%%

objects[2].shape


#%%    ########## with resize   #########

import os
import cv2
import numpy as np

def paste_object(dest_img_path, cropped_object, min_x, min_y, max_x, max_y, resize_w=None, resize_h=None):
    # Load the destination image
    dest_image = cv2.imread(dest_img_path, cv2.IMREAD_UNCHANGED)
    dest_h, dest_w = dest_image.shape[:2]

    # Calculate the position in the destination image
    x = int(min_x * dest_w)
    y = int(min_y * dest_h)
    max_x = int(max_x * dest_w)
    max_y = int(max_y * dest_h)

    # Resize the cropped object if resize dimensions are provided
    if resize_w and resize_h:
        obj_h, obj_w = cropped_object.shape[:2]
        aspect_ratio = obj_w / obj_h
        if resize_w / resize_h > aspect_ratio:
            resize_w = int(resize_h * aspect_ratio)
        else:
            resize_h = int(resize_w / aspect_ratio)
        resized_object = cv2.resize(cropped_object, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    else:
        resized_object = cropped_object

    # Ensure the resized object fits within the specified area
    obj_h, obj_w = resized_object.shape[:2]
    if obj_w > (max_x - x) or obj_h > (max_y - y):
        scale_x = (max_x - x) / obj_w
        scale_y = (max_y - y) / obj_h
        scale = min(scale_x, scale_y)
        new_w = int(obj_w * scale)
        new_h = int(obj_h * scale)
        resized_object = cv2.resize(resized_object, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a mask for the resized object
    mask = resized_object[:, :, 3]
    mask_inv = cv2.bitwise_not(mask)
    resized_object = resized_object[:, :, :3]

    # Define the region of interest (ROI) in the destination image
    roi = dest_image[y:y+resized_object.shape[0], x:x+resized_object.shape[1]]

    # Black-out the area of the object in the ROI
    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of the object from the object image
    obj_fg = cv2.bitwise_and(resized_object, resized_object, mask=mask)

    # Put the object in the ROI and modify the destination image
    dst = cv2.add(img_bg, obj_fg)
    dest_image[y:y+resized_object.shape[0], x:x+resized_object.shape[1]] = dst

    # Calculate the bounding box
    bbox = [x, y, resized_object.shape[1], resized_object.shape[0]]

    # Calculate the segmentation
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        segmentation.append(contour)

    return dest_image, bbox, segmentation

def create_coco_annotation(image_id, bbox, segmentation):
    annotation = {
        "image_id": image_id,
        "bbox": bbox,
        "segmentation": segmentation,
        "category_id": 1,  # Assuming a single category for simplicity
        "id": 1  # Annotation ID
    }
    return annotation

def export_coco_annotation(annotation, output_path):
    with open(output_path, 'w') as f:
        json.dump(annotation, f, indent=4)
#%% Example usage
#cropped_object = cv2.imread('path_to_cropped_object.png', cv2.IMREAD_UNCHANGED)
dest_img_path = '/home/lin/codebase/cv_with_roboflow_data/images/166.jpg'
min_x, min_y = 0.7, 0.7  # Define the minimum coordinates (0 to 1)
max_x, max_y = 0.999, 0.999  # Define the maximum coordinates (0 to 1)
resize_w, resize_h = 150, 150  # Define the resize dimensions for the cropped object

result_image, bbox, segmentation = paste_object(dest_img_path, objects[0], min_x, min_y, max_x, max_y, resize_w, resize_h)
cv2.imwrite('path_to_result_image.png', result_image)

annotation = create_coco_annotation(image_id=1, bbox=bbox, segmentation=segmentation)
export_coco_annotation(annotation, 'path_to_annotation.json')

#%%

# TODO object based clustering
# function take coco_annotation and img dir and does OB clustering
# function takes annotations and returns croped objects for each image {"imgname": [croped_objs]}
# function takes cropped obects and return features -> {"imgname": array(features)}
# function takes features and does clustering -> {feature: cluster}

#%%
def get_objects_per_img(coco_annotation_file, img_dir):
    coco = COCO(annotation_file=coco_annotation_file)
    img_names = [obj["file_name"] for obj in coco.imgs.values()]
    img_objects = {}
    for imgname in img_names:
        img_objs = get_objects_keep_imgdim(imgname, coco, img_dir)
        if img_objs:
            img_objs = [cv2.cvtColor(obj, cv2.COLOR_RGBA2RGB) for obj in img_objs]
            img_objects[imgname] = img_objs
        
    return img_objects
    
def get_obj_features_per_img(img_objects,img_resize_width,
                            img_resize_height,
                            model_family, model_name,
                            img_normalization_weight,
                            seed):
    img_feature = {}
    for imgname, objs in img_objects.items():
        feature = get_object_features(obj_imgs=objs, 
                                    img_resize_width=img_resize_width,
                                    img_resize_height=img_resize_height,
                                    model_family=model_family, model_name=model_name,
                                    img_normalization_weight=img_normalization_weight,
                                    seed=seed,
                                    )
        img_feature[imgname] = feature
    return img_feature



def cluster_img_features(img_feature: Dict) -> pd.DataFrame:
    img_names = [img_name for img_name in img_feature.keys()]
    img_feats = [feat for feat in img_feature.values()]
    featarray = np.array(img_feats)
    ce = clusteval()
    results = ce.fit(featarray)
    clusters = results["labx"]
    imgcluster_dict = {"image_names": img_names, "clusters": clusters}
    imgclust_df = pd.DataFrame.from_dict(imgcluster_dict)
    return imgclust_df
        

def object_based_cluster_images_from_cocoann(coco_annotation_file, img_dir,
                                             seed=2024, img_resize_width=224,
                                            img_resize_height=224,
                                            model_family="efficientnet",
                                            model_name="EfficientNetB0",
                                            img_normalization_weight="imagenet",
                                            ):
    
    img_objects = get_objects_per_img(coco_annotation_file=coco_annotation_file,
                                        img_dir=img_dir
                                        )
    img_feature = get_obj_features_per_img(img_objects=img_objects, 
                                           img_resize_width=img_resize_width,
                                            img_resize_height=img_resize_height,
                                            model_family=model_family,
                                            model_name=model_name,
                                            img_normalization_weight=img_normalization_weight,
                                            seed=seed
                                            )  
    cluster_df = cluster_img_features(img_feature=img_feature) 
    return cluster_df
#%%

img_objs = get_objects_per_img(coco_annotation_file=tomato_coco_path, img_dir=img_dir)


#%%

img_feats = get_obj_features_per_img(img_objects=img_objs, 
                                     img_resize_height=224,
                         img_resize_width=224,
                        model_family="efficientnet",
                        model_name="EfficientNetB0",
                        img_normalization_weight="imagenet",seed=2024
                         )
#%%
cluster_df = object_based_cluster_images_from_cocoann(coco_annotation_file=tomato_coco_path,
                                                      img_dir=img_dir
                                                      )
#%%




# %%
objects_in_img
# %%
for i in objects_in_img.keys():
    img_objects = objects_in_img[i]
    for obj in img_objects:
        #print(f"image: {i}")
        Image.fromarray(obj).show()
# %%
# 