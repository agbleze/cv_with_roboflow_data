
#%%
from typing import NamedTuple
import tensorflow as tf
import torch
from PIL import Image
import random
import numpy as np
import multiprocessing
from dataclasses import dataclass
from typing import List
from glob import glob
import os
from multiprocessing import Process, Lock




#%%
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.config.set_visible_devices([], 'GPU')

class ModelPreprocessReturn(NamedTuple):
    model: str
    preprocess: str
    
@dataclass
class ImgPropertySetReturnType:
    img_names: List
    img_paths: List
    total_num_imgs: int
    max_num_clusters: int    
    
def load_model_and_preprocess(input_shape, model_family, model_name, weight
                              ):
    model = getattr(tf.keras.applications, model_name)(input_shape=input_shape,
                                                      weights=weight,
                                                      include_top=False)
    preprocess = getattr(tf.keras.applications, model_family).preprocess_input
    model_preprocess_result = ModelPreprocessReturn(model=model, preprocess=preprocess)
    return model_preprocess_result

class FeatureExtractor(object):
    def __init__(self, seed=2024, img_resize_width=224,
                 img_resize_height=224, model_family="efficient",
                 model_name="EfficientNetB0",
                 img_normalization_weight="imagenet"
                 ):
        self.seed=seed
        self.img_resize_width=img_resize_width
        self.img_resize_height=img_resize_height
        self.model_family=model_family
        self.model_name=model_name
        self.image_shape=(img_resize_height, img_resize_width, 3)
        self.img_normalization_weight=img_normalization_weight
        
    def set_seed_consistently(self, seed=2024):
        if seed:
            self.seed=seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
    def load_and_resize_image(self, img_path, width=None, height=None):
        if not width:
            width = self.img_resize_width
        
        if not height:
            height = self.img_resize_height
            
        img = Image.open(img_path).resize((width, height))
        return img
    
    def load_model_and_preprocess_func(self, input_shape=None,
                                       model_family=None, model_name=None,
                                       weight=None
                                       ):
        if not input_shape:
            input_shape = self.image_shape
            
        if not model_family:
            model_family = self.model_family
        if not model_name:
            model_name = self.model_name
        if not weight:
            weight = self.img_normalization_weight
        
        model = getattr(tf.keras.applications, model_name)(input_shape=input_shape,
                                                      weights=weight,
                                                      include_top=False)
        preprocess = getattr(tf.keras.applications, model_family).preprocess_input
        self.model_preprocess_result = ModelPreprocessReturn(model=model, preprocess=preprocess)
        return self.model_preprocess_result
    
    def _check_model_preprocess_exist(self):
        if hasattr(self, "model_preprocess_result"):
            model, preprocess = self.model_preprocess_result
            return model, preprocess
        else:
            model, preprocess = self.load_model_and_preprocess_func()
            return model, preprocess
        
    def get_feature_extractor(self, model=None):
        if not model:
            model, preprocess = self._check_model_preprocess_exist()
            
        self.inputs = model.inputs
        x = model(self.inputs)
        outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
        self.feature_extractor = tf.keras.Model(inputs=self.inputs, outputs=outputs,
                                                name="feature_extractor")
        return self.feature_extractor
    
    def extract_features(self, inputs, model=None, preprocess=None):
        if not model or not preprocess:
            model, preprocess = self._check_model_preprocess_exist()
        x = preprocess(inputs)
        preds = model(x)
        return preds[0]
    
    def load_image_for_inference(self, image_path, img_shape=None):
        if not img_shape:
            img_shape = self.image_shape
        image = tf.io.read_file(image_path)
        x = tf.image.decode_image(image, channels=img_shape[2])
        x = tf.image.resize(x, (img_shape[0], img_shape[1]))
        x = tf.expand.dims(x, axis=0)
        return x
    
    def get_images_features(self, img_property_set, feature_extractor=None,
                            preprocess=None,
                            use_cropped_imgs=False,
                            use_merged_cropped_imgs=False
                            ):
        if not feature_extractor:
            if hasattr(self, "feature_extractor"):
                feature_extractor = self.feature_extractor
            else:
                feature_extractor = self.get_feature_extractor()
                
        if not preprocess:
            model, preprocess = self._check_model_preprocess_exist()
            
        images = []
        features = []
        
        if use_cropped_imgs and use_merged_cropped_imgs:
            raise Exception("both not provided")
        
        if use_merged_cropped_imgs:
            img_paths = sorted(img_property_set.merged_cropped_img_paths)
        if use_cropped_imgs:
            img_paths = sorted(img_property_set.cropped_img_paths)
        else:
            img_paths = sorted(img_property_set.img_paths)
            
        if use_cropped_imgs:
            img_property_set.cropped_imgs = images
            img_property_set.cropped_img_features = features
            
        img_property_set.imgs = images
        img_property_set.features = features
        
        
def get_imgs_and_extract_features_multiprocess(img_path, img_resize_width,
                                               img_resize_height,
                                               model_family, model_name,
                                               img_normalization_weight,
                                               seed, images_list, features_list, 
                                               #model_artefacts_dict, #lock
                                               ):
    #lock.acquire()
    feat_extract = FeatureExtractor(seed=seed, img_resize_width=img_resize_width,
                                    img_resize_height=img_resize_height, 
                                    model_family=model_family,
                                    model_name=model_name,
                                    img_normalization_weight=img_normalization_weight
                                    )
    images_list = []
    features_list = []
    feat_extract.set_seed_consistently()
    #model = model_artefacts_dict['model']
    #preprocess = model_artefacts_dict['preprocess']
    model, preprocess = feat_extract.load_model_and_preprocess_func()
    feature_extractor = feat_extract.get_feature_extractor(model)
    img = feat_extract.load_and_resize_image(img_path, img_resize_width, img_resize_height)
    img_for_infer = feat_extract.load_image_for_inference(img_path, feat_extract.image_shape)
    feature = feat_extract.extract_features(img_for_infer, feature_extractor, preprocess)
    print(f"feature: {feature} \n")
    images_list.append(img)
    features_list.append(feature)
    print(f"total imgs processed {len(images_list)}")
    print(f"total features processed {len(features_list)}")
    #lock.release()
    return images_list, features_list



def img_feature_extraction_implementor(img_property_set,
                                       feature_extractor_class = None,
                                       seed=2024, img_resize_width=224,
                                       img_resize_height=224,
                                       model_family="efficientnet",
                                       model_name="EfficientNetB0",
                                       img_normalization_weight="imagenet",
                                       use_cropped_imgs=True,
                                       ):
    img_paths = sorted(img_property_set.img_paths)
    
    manager = multiprocessing.Manager()
    images_list = manager.list()
    features_list = manager.list()
    #model_artefacts_dict = manager.dict()
    
    image_shape=(img_resize_height, img_resize_width, 3)
    
    #model, preprocess = load_model_and_preprocess(input_shape=image_shape, model_family=model_family, 
    #                                                model_name=model_name, 
    #                                                weight=img_normalization_weight,
    #                                                )
    #model_artefacts_dict['model'] = model
    #model_artefacts_dict['preprocess'] = preprocess
    #print(type(model_artefacts_dict))
    args_for_multiprocess = [(img_path, img_resize_width, img_resize_height,
                              model_family, model_name, 
                              img_normalization_weight, seed, images_list,
                              features_list, #model_artefacts_dict, #lock
                              )
                             for img_path in img_paths
                             ]
    num_processes = multiprocessing.cpu_count()
    
    with multiprocessing.Pool(num_processes) as pool:
    
        print("waiting for multiprocess to finish")
        results = pool.starmap(get_imgs_and_extract_features_multiprocess, args_for_multiprocess)
        
    images_list_re, features_list_re = results
    
    img_property_set.imgs = list(images_list_re)
    img_property_set.features = list(features_list_re)
    
    print(f"num of images: {len(img_property_set.imgs)}")
    print(f"num of features: {len(img_property_set.features)}")
    
    return img_property_set
    
    

#%%
img_dir = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/field_crop_with_disease"
img_paths_list = sorted(glob(f"{img_dir}/*"))   

img_property_set = ImgPropertySetReturnType(img_paths=img_paths_list, img_names="xxx", total_num_imgs=100, max_num_clusters=4)

if __name__ == '__main__':
    img_feature_extraction_implementor(img_property_set=img_property_set,
                                   use_cropped_imgs=False
                                   )
# %%
