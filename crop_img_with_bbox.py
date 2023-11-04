
#%%
from PIL import Image
import json
from typing import Callable, Union
import os
import pandas as pd
from pandas import json_normalize
from typing import NamedTuple, List, Union,Callable, Optional
from dataclasses import dataclass

#%%

valid_annot_file = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/valid/_annotations.coco.json"
# with open(valid_annot_file, "r") as annot_file:
#     annotation = json.load(annot_file)
    
#     for ann in annotation['annotations']:
#         img = Image.open(ann['file_name'])
#         x, y, w, h = ann['bbox']
#         cropped_img = img.crop((x,y,x+w, y+h))
#         cropped_img.save(f"{ann['id']}.jpg")


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
    all_merged_df.dropna(subset=["file_name"], inplace=True)
    return all_merged_df
    

#%%


annot_df = coco_annotation_to_df(coco_annotation_file=valid_annot_file)

#%%
annot_df['file_name'].values


#%%

annot_df
#%%
img_name = 'aphids-crop_jpg.rf.fdc584f5ace70e449ec59232d08e17ed.jpg'

test_img_item_df = annot_df[annot_df['file_name']==img_name]
test_img_item_bbox = test_img_item_df['bbox']

#%%
annot_df['bbox'].values
#%%
annot_df.dropna(subset=["file_name"])

#%%
def crop_image_with_bbox(coco_annotation_file_path: str, images_root_path: str,
                         output_dir: str,
                         all_images: bool = True,
                         image_name: Union[str, None] = None, 
                         ):
    annotation_record_df = coco_annotation_to_df(coco_annotation_file=coco_annotation_file_path)
    if all_images:
        for img in annotation_record_df['file_name'].values:
            img_df = annotation_record_df[annotation_record_df["file_name"]==img]
            for img_item in img_df['file_name'].values:
                img_item_df = img_df[img_df['file_name']==img_item]
                img_item_bbox = img_item_df['bbox'].to_list()[0]#.values
                x, y, w, h = img_item_bbox
                img_path = os.path.join(images_root_path, img_item)   
                img = Image.open(img_path)
                cropped_img = img.crop((x,y,x+w, y+h))
                ann_id = img_item_df['id_annotation'].to_list()[0]
                img_saved_name = f"{ann_id}_resized_{img_item}"
                img_ouput_path = os.path.join(output_dir, img_saved_name)
                cropped_img.save(img_ouput_path)
                #print(f"Successfully and cropped {img_item} with bbox {img_item_bbox} and saved as {img_saved_name}")
            

#%%
img_root = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/valid"
    
crop_image_with_bbox(coco_annotation_file_path=valid_annot_file, 
                     images_root_path=img_root, output_dir="cropped_imgs"
                     )    
    



#%% func to get object forimgs in a folder

from glob import glob

subset_img_path = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/subset_extract_folder/valid_subset"

img_name_list = []
for img_path in sorted(glob(f"{subset_img_path}/*.jpg")):
    img_name_list.append(img_path.split("/")[-1])



#%%

subset_annot_df = annot_df[annot_df['file_name'].isin(img_name_list)]

#%%

subset_annot_df.dropna(subset=['category_name', "id_annotation"], inplace=True)

#%%
#subset_annot_df.file_name.values

subset_img_name_list = []
for file_name in img_name_list:
    if file_name in subset_annot_df.file_name.values:
        subset_img_name_list.append(file_name)


#%%
len(subset_img_name_list)
        
#%%
import pandas as pd
img_name_eg = "101Apple_Mosaic_jpg.rf.89074173e29639bf88ce6510ede55b3f.jpg"

subset_wider_df = pd.pivot(subset_annot_df, index="file_name", columns="id_annotation", values="category_name" ).reset_index()#.columns


#%%
img_labels = subset_wider_df[subset_wider_df['file_name'] == img_name_eg].dropna(axis=1).to_numpy()[0][1:-1].tolist()#[0]

img_labels

#%%

img_labels_list = []
img_with_labels_name_list = []

for img in subset_img_name_list:
    img_lab = subset_wider_df[subset_wider_df['file_name'] == img].dropna(axis=1).to_numpy()[0][1:-1].tolist()
    if len(img_lab) != 0:
        img_with_labels_name_list.append(img)
        img_labels_list.append(img_lab)



@dataclass
class ImgPropertySetReturnType:
    img_names: List
    img_labels: List
    img_paths: List

#%%
def get_img_names_labels_paths(img_dir, annot_records_df, img_ext: str = ".jpg"):

    """_summary_

    Args:
        img_dir (_type_): _description_
        img_ext (_type_): _description_
        annot_records_df (_type_): _description_

    Returns:
        _type_: Returns image name and label in pairs 
    """
    annot_record_wideformat_df = pd.pivot(annot_records_df, index="file_name", 
                               columns="id_annotation", 
                               values="category_name" ).reset_index()#.columns

    img_name_list = []
    img_label_list = []
    img_paths_list = sorted(glob(f"{img_dir}/*{img_ext}"))
    for img_path in img_paths_list:
        img_name_list.append(img_path.split("/")[-1])
        
    for img in img_name_list:
        img_label = annot_record_wideformat_df[annot_record_wideformat_df['file_name'] == img].dropna(axis=1).to_numpy()[0][1:-1].tolist()
        if len(img_label) == 0:
            img_label = "None"
            #img_name_list.append(img)
            img_labels_list.append(img_label)
        else:
            #img_name_list.append(img)
            img_labels_list.append(img_label)
    return ImgPropertySetReturnType(img_names=img_name_list, 
                                    img_labels=img_labels_list,
                                    img_paths=img_paths_list
                                    ) 

    
    
#%%

img_names_labels_paths = get_img_names_labels_paths(img_dir=subset_img_path, annot_records_df=annot_df,
                        ) 


#%%
for nm, lab, path in zip(img_names_labels_paths.img_names,
    img_names_labels_paths.img_labels,
    img_names_labels_paths.img_paths):
    print(f"{i for i in lab}")   
    
#%%
for labels in img_names_labels_paths.img_labels:
    print(f"{i for i in labels}")    
    
#%%
len(img_labels_list)

#%%
len(img_with_labels_name_list)

#%%


with open("example.tsv", "w") as eg_file:
    [eg_file.write(f"{i}    ") for i in img_labels]


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


