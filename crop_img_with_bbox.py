
#%%
from PIL import Image
import json
from typing import Callable, Union
import os

#%%

valid_annot_file = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/valid/_annotations.coco.json"
with open(valid_annot_file, "r") as annot_file:
    annotation = json.load(annot_file)
    
    for ann in annotation['annotations']:
        img = Image.open(ann['file_name'])
        x, y, w, h = ann['bbox']
        cropped_img = img.crop((x,y,x+w, y+h))
        cropped_img.save(f"{ann['id']}.jpg")


#%%
def coco_annotation_to_df(coco_annotation_file):
    with open(coco_annotation_file, "r") as annot_file:
        annotation = json.load(annot_file)
    annotations_df = json_normalize(annotation, "annotations")
    annot_imgs_df = json_normalize(annotation, "images")
    annot_cat_df = json_normalize(annotation, "categories")
    annotations_images_merge_df = annotations_df.merge(annot_imgs_df, left_on='image_id', 
                                                        right_on='id',
                                                        suffixes=("_annotation", "_image"),
                                                        how="outer"
                                                        )
    annotations_imgs_cat_merge = annotations_images_merge_df.merge(annot_cat_df, left_on="category_id", right_on="id",
                                                                    suffixes=(None, '_categories'),
                                                                    how="outer"
                                                                    )
    all_merged_df = annotations_imgs_cat_merge[['id_annotation', 'image_id','category_id', 'bbox', 'area', 'segmentation', 'iscrowd',
                                'file_name', 'height', 'width', 'name', 'supercategory'
                                ]]
    all_merged_df.rename(columns={"name": "category_name",
                                  "height": "image_height",
                                  "width": "image_width"}, 
                         inplace=True
                         )
    return all_merged_df
    

#%%


annot_df = coco_annotation_to_df(coco_annotation_file=valid_annot_file)

#%%
annot_df['file_name'].values

#%%
def crop_image_with_bbox(coco_annotation_file_path: str, images_root_path: str,
                         all_images: bool = True,
                         image_name: Union[str, None] = None, 
                         ):
    annotation_record_df = coco_annotation_to_df(coco_annotation_file=coco_annotation_file_path)
    if all_images:
        for img in annotation_record_df['file_name'].values:
            img_df = annotation_record_df[annotation_record_df["file_name"]==img]
            for img_item in img_df['file_name'].values:
                img_item_df = img_df[img_df['file_name']==img_item]
                img_item_bbox = img_item_df['bbox']#.values
                x, y, w, h = img_item_bbox
                img_path = os.path.join(images_root_path, img_item)   
                img = Image.open(img_path)
                cropped_img = img.crop((x,y,x+w, y+h))
                ann_id = img_item_df['id_annotation'].values
                img_saved_name = f"{ann_id}_resized_{img_item}"
                cropped_img.save(img_saved_name)
                print(f"Successfully and cropped {img_item} with bbox {img_item_bbox} and saved as {img_saved_name}")
            

#%%
img_root = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/valid"
    
crop_image_with_bbox(coco_annotation_file_path=valid_annot_file, 
                     images_root_path=img_root
                     )    
    

#%%


# %%
annotation.keys()
# %%
annotation['annotations'][0]


#%%
annotation['images'][10]

#%%
annotation['categories']


# %%
from pandas import json_normalize

#%%
annotations_df = json_normalize(annotation, "annotations")#.head()

#%%
annot_imgs_df = json_normalize(annotation, "images")#.head()

#%%
annot_cat_df = json_normalize(annotation, "categories")#.head()

#%%
annot_cat_df.head()

#%%
annotations_df['image_id'].nunique()

#%%
annot_imgs_df['id'].nunique()

#%%
#annot_imgs_df['image_id'] = annot_imgs_df['id']

annot_imgs_df

#%%

annotations_images_merge_df = annotations_df.merge(annot_imgs_df, left_on='image_id', 
                                                    right_on='id',
                                                    suffixes=("_annotation", "_image"),
                                                    how="outer"
                                                    )

#%%
annotations_images_merge_df.columns

#%%
annotations_imgs_cat_merge = annotations_images_merge_df.merge(annot_cat_df, left_on="category_id", right_on="id",
                                                                suffixes=(None, '_categories'),
                                                                how="outer"
                                                                )

#%%
annotations_imgs_cat_merge.columns

#%%
all_merged_df = annotations_imgs_cat_merge[['id_annotation', 'image_id','category_id', 'bbox', 'area', 'segmentation', 'iscrowd',
                            'file_name', 'height', 'width', 'name', 'supercategory'
                            ]]
#%%
all_merged_df

#%%
all_merge_json = all_merged_df.to_json()

all_merge_json


#%%

all_merge_json

#%%

all_merged_df.to_dict() #.to_records().item()
#%%
annotations_df['category_id'].nunique()


#%%






# %%
data = [{
        "state": "Florida",
         "shortname": "FL",
         "info": {"governor": "Rick Scott"},
         "counties": [
             {"name": "Dade", "population": 12345},
             {"name": "Broward", "population": 40000},
             {"name": "Palm Beach", "population": 60000},
         ],
     },
     {
         "state": "Ohio",
         "shortname": "OH",
         "info": {"governor": "John Kasich"},
         "counties": [
             {"name": "Summit", "population": 1234},
             {"name": "Cuyahoga", "population": 1337},
         ],
     },
 ]

#%%
result = json_normalize(data, "counties", ["state", "shortname", ["info", "governor"]]
)



# %%