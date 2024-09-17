import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import albumentations as A

# Load COCO annotations
coco = COCO('path/to/annotations.json')

# Get the image ID and corresponding annotations
image_id = 123  # Replace with your image ID
ann_ids = coco.getAnnIds(imgIds=image_id)
anns = coco.loadAnns(ann_ids)

# Load the image
image = cv2.imread('path/to/image.jpg')
height, width = image.shape[:2]

# Initialize an empty mask
masks = []
bboxes = []

# Create the binary masks and bounding boxes
for ann in anns:
    rle = maskUtils.frPyObjects(ann['segmentation'], height, width)
    m = maskUtils.decode(rle)
    masks.append(m)
    bboxes.append(ann['bbox'])

# Stack masks along the third dimension
masks = np.stack(masks, axis=-1)

# Define Albumentations transform
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# Apply the transformation
transformed = transform(image=image, masks=masks, bboxes=bboxes, category_ids=[ann['category_id'] for ann in anns])
transformed_image = transformed['image']
transformed_masks = transformed['masks']
transformed_bboxes = transformed['bboxes']
transformed_category_ids = transformed['category_ids']

# Convert augmented masks and bounding boxes back to COCO format
new_annotations = []
for i, mask in enumerate(transformed_masks):
    rle = maskUtils.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string
    bbox = transformed_bboxes[i]
    category_id = transformed_category_ids[i]
    
    new_annotations.append({
        'segmentation': rle,
        'area': float(np.sum(mask)),
        'iscrowd': 0,
        'image_id': image_id,
        'bbox': bbox,
        'category_id': category_id,
        'id': ann_ids[i]
    })

print(new_annotations)
