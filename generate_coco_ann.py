
#%%
import json
import random
import os

def generate_random_bbox(image_width, image_height):
    x = random.randint(0, image_width - 1)
    y = random.randint(0, image_height - 1)
    width = random.randint(1, image_width - x)
    height = random.randint(1, image_height - y)
    return [x, y, width, height]

def generate_random_segmentation(bbox):
    x, y, width, height = bbox
    points = [
        x, y,
        x + width, y,
        x + width, y + height,
        x, y + height
    ]
    return [points]

def create_coco_annotation(image_id, category_id, bbox, segmentation, ann_id):
    annotation = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "segmentation": segmentation,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0
    }
    return annotation

def generate_coco_annotation_file(image_width, image_height, output_path, img_list):
    images, annotations, categories = [], [], []
    if not img_list:
        raise ValueError(f"img_list is required to be a list of str or path but {img_list} was given")
    for idx, img_path in enumerate(img_list):
        category_id = random.sample(population=[1,2,3], k=1)[0]
        image_id = idx + 1
        bbox = generate_random_bbox(image_width, image_height)
        segmentation = generate_random_segmentation(bbox)
        annotation = create_coco_annotation(image_id, category_id, bbox, segmentation,
                                            ann_id=image_id
                                            )
        img_info = {"id": image_id,
                    "width": image_width,
                    "height": image_height,
                    "file_name": os.path.basename(img_path)
                }
        category_info = {"id": category_id,
                        "name": f"object_{category_id}",
                        "supercategory": "none"
                        }
        images.append(img_info)
        categories.append(category_info)
        annotations.append(annotation)
        
    coco_format = {"images": images,
                    "annotations": annotations,
                    "categories": categories
                    }
    
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=4)
