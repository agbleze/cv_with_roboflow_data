

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
            
def compose_albumentation_pipeline(augconfig, replay=False, 
                                   label_fields=[#"class_labels", 
                                                 "category_ids", 
                                                 "ann_ids"
                                                 ]
                                   ):
    is_valid_albumentation_parameter(augconfig=augconfig)
    pipeline = []
    for augtype, aug_params in augconfig.items():
        aug_func = getattr(A, augtype)
        pipeline.append(aug_func(**aug_params))
    if not replay:
        return A.Compose(pipeline, 
                        bbox_params=A.BboxParams(format="coco", 
                                                label_fields=label_fields
                                                )
                        )
    else:
        return A.ReplayCompose(transforms=pipeline,
                                bbox_params=A.BboxParams(format="coco", 
                                                        label_fields=label_fields
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
albu_compose = compose_albumentation_pipeline(augconfig=augconfig)


#%%
transformed = albu_compose(image=img, bboxes=bbox, 
             masks=segms, class_labels=cat, 
             class_categories=["ripe", "ripe", "ripe"]
             )


#%%

#Image.open(img_path)

albu_compose(image=Image.open(img_path), bboxes=bbox, 
             masks=segms, class_labels=cat, 
             class_categories=["ripe", "ripe", "ripe"]
             )

#%%

augimg = transformed["image"] 

#%%

augimg.shape
#%%

augimg.save()
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
from PIL import Image
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

def augment_and_visualize(augconfig, img_path, coco=coco):
    albu_compose = compose_albumentation_pipeline(augconfig=augconfig)
    img, anns = get_image_and_annotations(coco=coco, img_id=1)
    img_name = img["file_name"]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #for ann in anns:
    segms = [coco.annToMask(ann) for ann in anns]
    bbox = [ann["bbox"] for ann in anns]
    cat = [ann["category_id"] for ann in anns]
    ann_ids = [ann["id"] for ann in anns]
    class_categories = []
    for cat_id in cat:
        for _cat_id, cat_info in coco.cats.items():
            if _cat_id == cat_id:
                name = cat_info["name"]
                class_categories.append(name)
                
        
    bbox_converted = [[bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]] for bb in bbox]
    bbox_converted = torch.tensor(bbox_converted, dtype=torch.float)
    segms_tensor = torch.tensor(segms, dtype=torch.bool)
    
    transformed = albu_compose(image=img, bboxes=bbox, 
                                masks=segms, class_labels=cat, 
                                class_categories=["ripe", "ripe", "ripe"],
                                ann_ids = ann_ids
                                )
    augmented_image = transformed["image"] 
    filepath = "augmented_img.png"
    Image.fromarray(augmented_image).save(filepath)
    augmented_bbox = transformed["bboxes"] 
    augmented_masks = transformed["masks"]
    #F.to_pil_image(augmented_image)
    #augmented_image = F.to_tensor(F.to_pil_image(augmented_image)).to(torch.uint8)
    augmented_bbox_converted = [[bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]] for bb in augmented_bbox]
    augmented_bbox_converted = torch.tensor(augmented_bbox_converted, dtype=torch.float)
    augmented_segms = torch.tensor(augmented_masks, dtype=torch.bool)
    
    image = read_image(img_path)
    augmented_image = read_image(filepath)
    
    images_with_bboxes = draw_bounding_boxes(image=image, boxes=bbox_converted,
                                            colors="red", width=2
                                            )
    image_with_mask = draw_segmentation_masks(image=images_with_bboxes, 
                                              masks=segms_tensor, 
                                              alpha=0.9
                                              )
    
    
    
    augmentedimages_with_bboxes = draw_bounding_boxes(image=augmented_image, 
                                                      boxes=augmented_bbox_converted,
                                                        colors="red", width=2
                                                        )
    augmented_img_with_masks = draw_segmentation_masks(image=augmentedimages_with_bboxes,
                                                       masks=augmented_segms
                                                       )
    plt.imshow(F.to_pil_image(image_with_mask))
    plt.axis("off")
    plt.title("Original Image")
    plt.show()
    #Image
    #torch.asarray(augmented_img_with_masks)
    #print(augmented_img_with_masks.reshape((1,2,0)))
    augshape = augmented_img_with_masks.shape
    plt.imshow(F.to_pil_image(augmented_img_with_masks))
    #plt.imshow(augmented_img_with_masks.transpose(1,2,0))
    plt.axis("off")
    plt.title("Augmented Image")
    plt.show()
    
#%%
augment_and_visualize(augconfig=augconfig, img_path=img_path)   
    
#%%  TODO: use of ReplayCompose to record, serialize aug used
# for each image save the aug used wuth Replay

#%%
from glob import glob
import os
def augment_imgs(augconfig, img_dir, coco_annfilepath, 
                 save_augmented_annotation_as="augmented_annotations.json",
                 augmented_output_imgdir="augmented_imgs"
                 ):
    os.makedirs(augmented_output_imgdir, exist_ok=True)
    coco = COCO(coco_annfilepath)
    imglist = glob(f"{img_dir}/*")
    augmented_images_info = []
    augmented_annotation_info = []
    for img_path in imglist:
        imgname = os.path.basename(img_path)
        imgid = [coco_obj["id"] for _, coco_obj in coco.imgs.items() if 
                 os.path.basename(coco_obj["file_name"]) == imgname][0]

        ann_ids = coco.getAnnIds(imgIds=imgid)
        anns = coco.loadAnns(ids=ann_ids)
        segmasks = [coco.annToMask(ann) for ann in anns]
        all_annid = [ann["id"] for ann in anns]
        ann_category_ids = [ann["category_id"] for ann in anns]
        bboxes = [ann["bbox"] for ann in anns]
        augpipeline = compose_albumentation_pipeline(augconfig=augconfig)
        imgread = cv2.imread(img_path)
        image = cv2.cvtColor(src=imgread, code=cv2.COLOR_BGR2RGB)
        # include annotation ids
        augmented_obj = augpipeline(image=image, bboxes=bboxes, 
                                    masks=segmasks, 
                                    category_ids =ann_category_ids,
                                    ann_ids=all_annid
                                    )
        aug_image = augmented_obj["image"]
        aug_bboxes = augmented_obj["bboxes"]
        aug_masks = augmented_obj["masks"]
        aug_ann_ids = augmented_obj["ann_ids"]
        aug_category_ids = augmented_obj["category_ids"]
        
        aug_img_height, aug_img_width = aug_image.shape[0], aug_image.shape[1]
        # Calculate the segmentation
        segmentations = []
        for mask in aug_masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []
            for contour in contours:
                contour = contour.flatten().tolist()
                segmentation.append(contour)
            segmentations.append(segmentation)
            
        for img_info in coco.imgs.values():
            if os.path.basename(img_info["file_name"]) == imgname:
                img_info["width"] = aug_img_width
                img_info["height"] = aug_img_height
                augmented_images_info.append(img_info)
                
        # need to ensure correct order and number of annotations when spatial augmentation is 
        # performed and some of bbox are below min_area and min_visibility
        for aug_annid, aug_segmask, aug_bbox in zip(aug_ann_ids, segmentations, aug_bboxes):
            for anninfo in anns:
                if aug_annid == anninfo["id"]:
                    anninfo["bbox"] = aug_bbox
                    anninfo["segmentation"] = aug_segmask
                    area = aug_bbox[2] * aug_bbox[3]
                    anninfo["area"] = area
                    augmented_annotation_info.append(anninfo)
        output_augimg_path = os.path.join(augmented_output_imgdir, imgname)
        Image.fromarray(aug_image).save(output_augimg_path)
                    
        
                    
    with open(coco_annfilepath, "r") as filepath:
        cocodata = json.load(filepath)
        
        cocodata["images"] = augmented_images_info
        cocodata["annotations"] = augmented_annotation_info 
        
    with open(save_augmented_annotation_as, "w") as outfile:
        json.dump(cocodata, outfile)
        
    # TODO. visualization of augmented images
    
        
#%%
img_dir = "/home/lin/codebase/cv_with_roboflow_data/subset_extract_folder/tomato_fruit"
augment_imgs(augconfig=augconfig, img_dir=img_dir, coco_annfilepath=coco_path)
      
        
    
   
    
    
    


# %%
import numpy as np
from pycocotools import mask as coco_mask
from skimage import measure

def mask_to_polygon(binary_mask):
    # Ensure the mask is in Fortran order
    fortran_mask = np.asfortranarray(binary_mask)
    encoded_mask = coco_mask.encode(fortran_mask)
    contours = measure.find_contours(binary_mask, 0.5)

    polygons = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        polygons.append(segmentation)

    return polygons

# Example usage
binary_mask = np.array([[0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0]], dtype=np.uint8)

polygons = mask_to_polygon(binary_mask)
print(polygons)

# %%
# Import Supervision
import supervision as sv

# Import mask data (optional if you have raw mask data)
#detections = sv.Detections.from_inference(prediction)

# Convert each mask to a polygon
polygons = [sv.mask_to_polygons(binary_mask)]
# for raw mask data: polygons = sv.mask_to_polygons(mask)
print(polygons)

# %%
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import rembg
import io

def change_background_to_white(input_path, output_path):
    # Remove the background
    with open(input_path, "rb") as input_file:
        input_data = input_file.read()
        output_data = rembg.remove(input_data)

    # Open the image with the removed background
    img = Image.open(io.BytesIO(output_data)).convert("RGBA")

    # Create a white background image
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))

    # Composite the image with the white background
    final_img = Image.alpha_composite(white_bg, img)

    # Apply Gaussian blur to smooth the edges
    final_img = final_img.filter(ImageFilter.GaussianBlur(radius=2))

    # Convert to RGB and save the image
    final_img = final_img.convert("RGB")
    final_img.save(output_path, format='JPEG')
# Example usage
input_path = '/home/lin/codebase/cv_with_roboflow_data/1726908281123.jpg'
output_path = '/home/lin/codebase/cv_with_roboflow_data/output_image.jpg'
change_background_to_white(input_path, output_path)

# %%
from PIL import Image

def format_photo(input_path, output_path):
    # Open the image file
    img = Image.open(input_path)

    # Resize the image to 600x600 pixels
    img = img.resize((600, 600), Image.LANCZOS)

    # Save the formatted image
    img.save(output_path, format='JPEG')


format_photo(input_path, "reduced_image.jpg")

# %%
Image.open(input_path).resize((600, 600), Image.LANCZOS).save("reduced_image.jpg")
# %%
