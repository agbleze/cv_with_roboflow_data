
#%%
from PIL import Image
import json

valid_annot_file = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/valid/_annotations.coco.json"
with open(valid_annot_file, "r") as annot_file:
    annotation = json.load(annot_file)
    
    for ann in annotation['annotations']:
        img = Image.open(ann['file_name'])
        x, y, w, h = ann['bbox']
        cropped_img = img.crop((x,y,x+w, y+h))
        cropped_img.save(f"{ann['id']}.jpg")




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
all_merged_df.to_json()

#%%
annotations_df['category_id'].nunique()




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
