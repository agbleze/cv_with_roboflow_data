

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
    return A.Compose(pipeline)

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

#for ann in anns:
segms = [ann["segmentation"] for ann in anns]
bbox = [ann["bbox"] for ann in anns]
cat = [ann["category_id"] for ann in anns]



#%%
albu_compose = compose_albumentation_pipeline(augconfig=augconfig)





#%%

is_valid_albumentation_parameter(augconfig=augconfig)      

#%%
inspect.signature(getattr(A,albu_methods[0])).parameters


# %%
