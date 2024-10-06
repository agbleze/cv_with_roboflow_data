
#%%
import cv2
import json
import os
from pathlib import Path

#%%
def visualize_bboxes(annotation_file, image_dir, output_dir):
    # Load COCO annotation file
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Create a dictionary to map category IDs to category names
    category_map = {category['id']: category['name'] for category in coco_data['categories']}

    # Iterate over each image in the annotation file
    for image_info in coco_data['images']:
        image_id = image_info['id']
        image_path = os.path.join(image_dir, image_info['file_name'])

        # Load the image
        image = cv2.imread(image_path)

        # Get annotations for this image
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

        # Draw bounding boxes on the image
        for ann in annotations:
            bbox = ann['bbox']
            category_id = ann['category_id']
            category_name = category_map[category_id]

            x, y, w, h = map(int, bbox)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the image with bounding boxes
        output_path = os.path.join(output_dir, f"bbox_{image_info['file_name']}")
        cv2.imwrite(output_path, image)
        print(f"Bounding boxes visualized for {image_info['file_name']} and saved as {output_path}")

#%% Example usage
if __name__ == '__main__':
    annotation_file = "/home/lin/codebase/merge_coco/cor4_merged_annotations.json"
    valid_coco = "/home/lin/codebase/merge_coco/valid_annotations.json"
    image_dir = "/home/lin/codebase/merge_coco/valid"
    output_dir = "/home/lin/codebase/merge_coco/valid_viz_cor4"
    visualize_bboxes(valid_coco, image_dir, output_dir)

# %%
# flea-beetle-2_jpg.rf.5d03f74e1f36cc80c7606c28f6589420
