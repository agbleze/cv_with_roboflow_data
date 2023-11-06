
from PIL import Image
import json
from typing import Callable, Union
import os
import pandas as pd
from pandas import json_normalize
from typing import NamedTuple, List, Union,Callable, Optional, Dict
from dataclasses import dataclass
from glob import glob
import cv2
import numpy as np



@dataclass
class ImgPropertySetReturnType:
    img_names: List
    img_labels: List
    img_paths: List



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
    

def merge_imgs(img_list, 
				interpolation = cv2.INTER_CUBIC,
                merge_type="horizontal"):
    img_array_list = [np.array(img) for img in img_list]

    if merge_type == "horizontal":
        h_min = min(img.shape[0] for img in img_array_list) 
        im_list_resize = [cv2.resize(img, 
					(int(img.shape[1] * h_min / img.shape[0]), 
						h_min), interpolation 
								= interpolation) 
					for img in img_array_list]
        return cv2.hconcat(im_list_resize)
    else:
        h_min = min(img.shape[1] for img in img_array_list)
        im_list_resize = [cv2.resize(img, 
                        (int(img.shape[0] * h_min / img.shape[1]), 
                            h_min), interpolation 
                                    = interpolation) 
                        for img in img_array_list] 
    return cv2.vconcat(im_list_resize)


def crop_image_with_bbox(images_root_path: str,
                         output_dir: str,
                         all_images: bool = True, 
                         coco_annotation_file_path: Optional[str] = None,
                         image_names: Union[str, List, None] = None, 
                         use_annotation_record_df: bool = False,
                         annotation_record_df: Union[pd.DataFrame, None] = None,
                         result_store_type: Union[ImgPropertySetReturnType, None] = ImgPropertySetReturnType,
                         save_img_ext: Union[str, None] = ".jpg",
                         export_merged_crops_per_img: bool = True,
                         merged_crops_output_dir: str = "merged_cropped_bbox",
                         merge_crop_of_imgs: bool = False,
                         )->Union[ImgPropertySetReturnType, Dict]:
    
    existing_imgs = glob(f"{output_dir}/*{save_img_ext}")
    print("Will remove existing images with {save_img_ext} in output directory if found")
    [os.remove(f) for f in existing_imgs if len(existing_imgs) > 0]
    os.makedirs(output_dir, exist_ok=True)
    if use_annotation_record_df and (not annotation_record_df or not isinstance(annotation_record_df, pd.DataFrame)):
        raise Error("""annotation_record_df is None or not a pandas DataFrame while use_annotation_record_df is True. 
                        Please provide a dataframe for annotation_record_df with a column name as file_name for 
                        image names OR set use_annotation_record_df to False and provide a coco annotation file path
                        for coco_annotation_file_path
                    """
                    )
    elif use_annotation_record_df:
        annotation_record_df = annotation_record_df
    
    else:
        try:
            annotation_record_df = coco_annotation_to_df(coco_annotation_file=coco_annotation_file_path)
        except Exception as e:
            raise("""coco_annotation_file_path not found. Provide the correct path or 
                  set use_annotation_record_df to True and provide provide a dataframe 
                  for annotation_record_df with a column name as file_name for image names
                """
                )
     
    if "file_name" not in annotation_record_df.columns:
            raise Error("""The is no file_name key in coco annotation file or no such column name
                        in the 
                        annotation_record_df provided. This is required"""
                        )       

    if all_images:
        list_of_image_names = sorted(annotation_record_df['file_name'].values)
    else:
        if isinstance(image_names, str):
            image_names = [image_names]
        
        else:
            if not isinstance(image_names, List):
                raise Error("Image names must be provided as a list if all_images is set to False")
        list_of_image_names = sorted(image_names)    
     
    cropped_img_path_list = [] 
    cropped_img_name_list = [] 
    cropped_img_labels_list = [] 
    merged_cropped_img_path_list = []
    merged_cropped_img_list = [] 
    #all_crops_per_img = []
    for img_name_item in list_of_image_names:
        crops_per_img = []
        img_df = annotation_record_df[annotation_record_df["file_name"]==img_name_item]
        for ann_id in img_df['id_annotation'].to_list():
            img_item_df = img_df[img_df['id_annotation']==ann_id]
            img_item_bbox = img_item_df['bbox'].to_list()[0]#.values
            x, y, w, h = img_item_bbox
            img_name = img_item_df['file_name'].to_list()[0]
            img_path = os.path.join(images_root_path, img_name)   
            img = Image.open(img_path)
            cropped_img = img.crop((x,y,x+w, y+h))
            
            if merge_crop_of_imgs:
                crops_per_img.append(cropped_img)
            
            #ann_id = img_item_df['id_annotation'].to_list()[0]
            # TODO: merge_crops
            #
            
            if save_img_ext:
                img_name_without_ext = os.path.splitext(img_name)[0]
                default_img_ext = os.path.splitext(img_name)[1]
                if default_img_ext != save_img_ext:
                    print(f"Default Image in {default_img_ext} but saving cropped img in {save_img_ext}")
                    
                    img_name = img_name_without_ext + save_img_ext
            img_saved_name = f"{ann_id}_cropped_{img_name}"
            img_ouput_path = os.path.join(output_dir, img_saved_name)
            cropped_img.save(img_ouput_path)
            
            
            
            ## cropped img labels
            img_item_label = img_item_df['category_name'].to_list()[0]#[img_item_df['id_annotation']==ann_id]
            if len(img_item_label) == 0:
                label = "None"
                cropped_img_labels_list.append([label])
            else:
                cropped_img_labels_list.append(img_item_label)    
            
            
            cropped_img_name_list.append(img_saved_name)
            cropped_img_path_list.append(img_ouput_path)
        
        if merge_crop_of_imgs:
            merged_img = merge_imgs(img_list=crops_per_img)
            merged_cropped_img_list.append(merged_img)
            if export_merged_crops_per_img:
                os.makedirs(merged_crops_output_dir, exist_ok=True)
                if export_merged_crops_per_img:
                    save_merged_crop_path = os.path.join(merged_crops_output_dir, img_name_item)
                    # show the output image 
                    cv2.imwrite(save_merged_crop_path, merged_img)
                    merged_cropped_img_path_list.append(save_merged_crop_path)
                    #np.save(img_name_item, merged_img)
                   # merged_img.save(img_name_item)
                
            
    if isinstance(result_store_type, ImgPropertySetReturnType):
        result_store_type.cropped_img_names = cropped_img_name_list
        result_store_type.cropped_img_paths = cropped_img_path_list
        result_store_type.cropped_img_labels = cropped_img_labels_list
        result_store_type.merged_cropped_img_paths = merged_cropped_img_path_list
        result_store_type.merged_cropped_imgs = merged_cropped_img_list
            
    else:
        result_store_type = {"cropped_img_names": list_of_image_name_list, 
                            "cropped_img_paths": cropped_img_path_list,
                            "cropped_img_labels": cropped_img_labels_list,
                            "merged_cropped_img_paths": merged_cropped_img_path_list,
                            "merged_cropped_imgs": merged_cropped_img_list
                            }
    return result_store_type

def get_img_names_labels_paths(img_dir: str, annot_records_df: pd.DataFrame, 
                               img_ext: str = ".jpg"
                               )-> ImgPropertySetReturnType:

    """_summary_

    Args:
        img_dir (_type_): _description_
        img_ext (_type_): _description_
        annot_records_df (_type_): _description_

    Returns:
        ImgPropertySetReturnType: Returns image name and label in pairs 
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
            img_label_list.append([img_label])
        else:
            #img_name_list.append(img)
            img_label_list.append(img_label)
    return ImgPropertySetReturnType(img_names=img_name_list, 
                                    img_labels=img_label_list,
                                    img_paths=img_paths_list
                                    ) 


