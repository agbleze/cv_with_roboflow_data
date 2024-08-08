
#%%
from pycocotools.coco import COCO
import cv2
import numpy as np
from PIL import Image
from clusteval import clusteval
import pandas as pd
import json
from feat import get_object_features
from typing import Union, List, Dict
#%% Load COCO annotations
coco = COCO('/home/lin/codebase/cv_with_roboflow_data/coco_annotation_coco.json')

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

#%%
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

def paste_object(dest_img_path, cropped_objects, min_x, min_y, max_x, max_y, resize_w=None, resize_h=None):
    # Load the destination image
    dest_image = cv2.imread(dest_img_path, cv2.IMREAD_UNCHANGED)
    dest_h, dest_w = dest_image.shape[:2]

    if not isinstance(cropped_objects, list):
        cropped_objects = [cropped_objects]
    # Calculate the position in the destination image
    x = int(min_x * dest_w)
    y = int(min_y * dest_h)
    max_x = int(max_x * dest_w)
    max_y = int(max_y * dest_h)
    bboxes, segmentations = [], []
    # Resize the cropped object if resize dimensions are provided
    for cropped_object in cropped_objects:
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
        bboxes.append(bbox)
        segmentations.append(segmentation)
    return dest_image, bboxes, segmentations

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
#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
import cv2
cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)


#%%
cv2.imwrite("/home/lin/codebase/cv_with_roboflow_data/viz_image.png", result_image)
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


import numpy as np
from pycocotools import mask

def annToMask(ann, height, width):
    """
    Convert annotation to binary mask.
    
    Parameters:
    ann (dict): COCO annotation dictionary.
    height (int): Height of the image.
    width (int): Width of the image.
    
    Returns:
    np.ndarray: Binary mask.
    """
    segmentation = ann['segmentation']
    rle = mask.frPyObjects(segmentation, height, width)
    binary_mask = mask.decode(rle)
    
    return binary_mask

#%%   #### TODO copy-paste augmentation
# get objects define from images -- cropping based on obj name
#####  have option to limit number of objects to crop when number of images are large
# cropped objects should be stored based on obj name separately
#  ---- pasting of objects ----
# sample from cropped obj list and paste into background obj
#     write annotation after pasting. 
# for each obj pasted give unique ann id and store in a list
##### increase ann id based on stored list to ensure they are unique
##### image_name: name of background image / image pasted into
##### image_id: count of background image
##### category and category id should be base on paste obj 
#### centralize the annotation format and update it 


#%%
from typing import Union
def crop_obj_per_image(obj_names: list, imgnames: Union[str, List], img_dir,
                       coco_ann_file: str
                       ) -> Union[Dict[str,List], None]:
    #cropped_objs_collection = {obj: [] for obj in obj_names}
    #print(f"cropped_objs_collection: {cropped_objs_collection} \n")
    cropped_objs_collection = {}
    # get objs in image
    with open(coco_ann_file, "r") as filepath:
        coco_data = json.load(filepath)
        
    categories = coco_data["categories"]
    category_id_to_name_map = {cat["id"]: cat["name"] for cat in categories}
    category_name_to_id_map = {cat["name"]: cat["id"] for cat in categories}
    
    if isinstance(imgnames, str):
        imgnames = [imgnames]
    images = coco_data["images"]
    for imgname in imgnames:
        image_info = [img_info for img_info in images if img_info["file_name"]==imgname][0]
        image_id = image_info["id"]
        image_height = image_info["height"]
        image_width = image_info["width"]
        annotations = coco_data["annotations"]
        img_ann = [ann_info for ann_info in annotations if ann_info["image_id"]==image_id]
        img_catids = set(ann_info["category_id"] for ann_info in img_ann)
        img_objnames = [category_id_to_name_map[catid] for catid in img_catids]
        img_path = os.path.join(img_dir, imgname)
        image = cv2.imread(img_path)
        objs_to_crop = set(img_objnames).intersection(set(obj_names))
        if objs_to_crop:
            for objname in obj_names:
                print(f"objname: {objname} \n")
                object_masks = []
                if objname in img_objnames:
                    obj_id = category_name_to_id_map[objname]
                    for ann in img_ann:
                        if ann["category_id"] == obj_id:
                            mask = annToMask(ann=ann, height=image_height, width=image_width)
                            
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for contour in contours:
                                x, y, w, h = cv2.boundingRect(contour)
                                cropped_object = image[y:y+h, x:x+w]
                                object_masks.append(cropped_object)
                                #print(f"in contours loop cropped_objs_collection[objname]: {cropped_objs_collection[objname]} \n")
                                #cropped_objs_collection[objname] = cropped_objs_collection[objname].append([cropped_object])
                print(f"imgname: {imgname},  objname: {objname}")
                if objname not in cropped_objs_collection.keys():
                    cropped_objs_collection[objname] = object_masks
                else:
                    for each_mask in object_masks:
                        cropped_objs_collection[objname] = cropped_objs_collection[objname].append(each_mask)
            
            
    return cropped_objs_collection
        #else:
        #    return None

#%%
coco_ann_path = "/home/lin/codebase/cv_with_roboflow_data/tomato_coco_annotation/annotations/instances_default.json"
img_dir= "/home/lin/codebase/cv_with_roboflow_data/images"
imgname = "10.jpg"
objnames = ["ripe", "unripe", "flowers"]

cropped_obj_collect = crop_obj_per_image(obj_names=objnames, imgnames=imgname, img_dir=img_dir,
                                        coco_ann_file=coco_ann_path
                                        )

#%%

len(cropped_obj_collect["unripe"])

#%%
cropped_obj_collect["unripe"][2]


#%%
imgnames_for_cropping = ["0.jpg", "1235.jpg", "494.jpg", "446.jpg", "10.jpg"]
all_crop_objects = crop_obj_per_image(obj_names=objnames, imgnames=imgnames_for_cropping, img_dir=img_dir,
                   coco_ann_file=coco_ann_path)



#%% # pseudo code
def collate_all_crops(object_to_cropped, imgnames_for_crop, img_dir,
                      coco_ann_file
                      ):
    all_crops = {obj: [] for obj in object_to_cropped}
    allimg_crops = []
    for img in imgnames_for_crop:
        crop_obj = crop_obj_per_image(obj_names=object_to_cropped, 
                                      imgname=img, 
                                    img_dir=img_dir,
                                    coco_ann_file=coco_ann_file
                                    )
        allimg_crops.append(crop_obj)
        
    if allimg_crops:
        for crop_res in allimg_crops:
            ######  CONTINUE FROM HERE ##############
        
        
        if crop_obj:
            for crop in crop_obj:
                #print(f"{crop}: {len(crop_obj[crop])} \n")
                crop_obj_maskslist = crop_obj[crop]
                for crop_mask in crop_obj_maskslist:
                    if all_crops[crop] is None:
                        all_crops[crop] = [crop_mask]
                        print(f"In none")
                    else:
                        all_crops[crop] = all_crops[crop].append(crop_mask)
                        print(f"outside none")
                    
    return all_crops


#%%

imgnames_for_cropping = ["0.jpg", "1235.jpg", "494.jpg", "446.jpg", "10.jpg"]

all_crop_objects = collate_all_crops(object_to_cropped=objnames, imgnames_for_crop=imgnames_for_cropping,
                                    img_dir=img_dir, coco_ann_file=coco_ann_path
                                    )


#%%

len(all_crop_objects["unripe"])
#%%
for obj in all_crop_objects:
    obj_crops = all_crop_objects[obj]
    if obj_crops is not None:
        for obj_crop in obj_crops:
            print(f"obj: {obj}")
            Image.fromarray(obj_crop).show()
         
#%%
# after collating all crops, sample number of objects to be cropped
# for each object and paste for each background image
import random
def paste_crops_on_bkgs(bkgs, all_crops, objs_paste_num: Dict, output_img_dir, save_coco_ann_as):
    os.makedirs(output_img_dir, exist_ok=True)
    coco_ann = {"categories": [], "images": [], "annotations": []}
    ann_ids = []
    for bkg_idx, bkg in enumerate(bkgs):
        for obj_idx, obj in enumerate(objs_paste_num):
            num_obj = objs_paste_num[obj]
            objs_to_paste = all_crops[obj]
            sampled_obj = random.sample(objs_to_paste, int(num_obj))
            dest_img, bboxes, segmasks = paste_object(dest_img_path=bkg,
                                                        cropped_objects=sampled_obj,
                                                        )
            file_name = os.path.basename(bkg)
            img_path = os.path.join(output_img_dir, file_name)
            cv2.imwrite(img_path, dest_img)
            assert(len(bboxes) == len(segmasks)), f"bboxes and segmasks are not equal length"
                       
            image = cv2.imread(bkg)
            img_height, img_width = image.shape[0], image.shape[1]
            img_id = bkg_idx+1
            
            
            image_info = {"file_name": file_name, "height": img_height, 
                          "width": img_width, "id": img_id
                          }
            obj_category = {"id": obj_idx + 1, "name": obj}
            coco_ann["categories"] = coco_ann["categories"].append(obj_category)
            coco_ann["images"] = coco_ann["images"].append(image_info)
            
            for ann_ins in range(0, len(bboxes)):
                bbox = bboxes[ann_ins]
                segmask = segmasks[ann_ins]
                ann_id = len(ann_ids) + 1
                annotation = {"id": ann_id, "image_id": img_id, 
                              "bbox": bbox,
                              "segmentation": segmask
                              } 
                coco_ann["annotations"] = coco_ann["annotations"].append(annotation)
    with open(save_coco_ann_as, "w") as filepath:
        json.dump(coco_ann, filepath)            
                
            
#%%
bkgs = ["/home/lin/codebase/cv_with_roboflow_data/images/1859.jpg",
        "/home/lin/codebase/cv_with_roboflow_data/images/1668.jpg",
        "/home/lin/codebase/cv_with_roboflow_data/images/1613.jpg",
        "/home/lin/codebase/cv_with_roboflow_data/images/1541.jpg",
        "/home/lin/codebase/cv_with_roboflow_data/images/1892.jpg"
        ]

obj_paste_num = {"ripe": 1, "unripe": 1}    
paste_crops_on_bkgs(bkgs=bkgs, all_crops=all_crop_objects, 
                    objs_paste_num=obj_paste_num,
                    output_img_dir="pasted_output_dir",
                    save_coco_ann_as="cpaug.json"
                    )


            
#%%

exmdict = {"first": [], "second": []}

if not exmdict["first"]:
    print("empty")
else:
    print("occupied")
# %%
import random

random.sample([1,2,3,4,9,2,3,3], 4)
# %%
