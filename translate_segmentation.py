#%%
import json
from pycocotools.coco import COCO

# Load COCO annotations
coco = COCO('path_to_annotation.json')

# Function to adjust segmentation based on bbox
def adjust_segmentation(bbox, segmentation):
    x_offset, y_offset = bbox[0], bbox[1]
    adjusted_segmentation = []
    for polygon in segmentation:
        adjusted_polygon = []
        for i in range(0, len(polygon), 2):
            adjusted_polygon.append(polygon[i] + x_offset)
            adjusted_polygon.append(polygon[i + 1] + y_offset)
        adjusted_segmentation.append(adjusted_polygon)
    return adjusted_segmentation

# Get annotations
annotations = coco.loadAnns(coco.getAnnIds())

# Adjust segmentation for each annotation
for ann in annotations:
    bbox = ann['bbox']
    segmentation = ann['segmentation']
    ann['segmentation'] = adjust_segmentation(bbox, segmentation)

# Save adjusted annotations
with open('adjusted_annotations.json', 'w') as f:
    json.dump(annotations, f)

# %%
import json
from pycocotools.coco import COCO

# Load COCO annotations
#coco = COCO('path_to_annotations.json')

# Function to adjust segmentation based on bbox
def adjust_segmentation(bbox, segmentation):
    x_offset, y_offset = bbox[0], bbox[1]
    adjusted_segmentation = []
    for polygon in segmentation:
        adjusted_polygon = []
        for i in range(0, len(polygon), 2):
            adjusted_polygon.append(polygon[i] + x_offset)
            adjusted_polygon.append(polygon[i + 1] + y_offset)
        adjusted_segmentation.append(adjusted_polygon)
    return adjusted_segmentation

# Example bbox and segmentation
bbox = [50, 100, 200, 150]
segmentation = [[0, 0, 50, 0, 50, 50, 0, 50]]

# Adjust segmentation
adjusted_segmentation = adjust_segmentation(bbox, segmentation)

print("Original Segmentation:", segmentation)
print("Adjusted Segmentation:", adjusted_segmentation)

# %%
