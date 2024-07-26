
from clustimage import Clustimage
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import lru_cache
from typing import Union, List, NamedTuple
import abc
import cv2
from zipfile import ZipFile
from glob import glob
import os
from dataclasses import dataclass
from sklearn.manifold import TSNE
import plotly.express as px


@dataclass
class FolderReturns:
    uploaded_file_name: str
    img_folder: str
    img_list: str
    
class UploadedFileUtil:
    def __init__(self, extract_folder_name="extract_folder"):
        self.extract_folder_name = extract_folder_name
   
    #@lru_cache(maxsize=None)
    def unzip_upload(self, filenames_list):
        self.filenames_list = filenames_list
        
        zip_file_1 = self.filenames_list[0]
        with ZipFile(zip_file_1, "r") as file:
                file.extractall(self.extract_folder_name)
                    
    def get_upload_paths_names(self, img_ext: str = ".jpg") -> FolderReturns:
        self.uploaded_file_name = self.filenames_list[0].split(".")[0]
        self.img_path = os.path.join(self.extract_folder_name, self.uploaded_file_name)
        self.imgs_path_list = glob(f"{self.img_path}/*{img_ext}")
        return FolderReturns(uploaded_file_name=self.uploaded_file_name,
                             img_folder=self.img_path,
                             img_list=self.imgs_path_list
                             )


class ImageFolder(abc.ABC):
    def __init__(self):
        pass
    @abc.abstractmethod
    def get_img_folder(self, *args, **kwargs):
        pass

class ImageFolderGetter(ImageFolder):
    def __init__(self):
        pass
    def get_img_folder(self, list_of_contents: Union[List,None] = None,
                   list_of_names: Union[List, None] = None,
                   img_folder_path: Union[str, None] = None
                   ):
        if img_folder_path:
            self.img_folder_path
        else:
            self.list_of_contents = list_of_contents
            self.list_of_names = list_of_names
            folder_name = list_of_names[0].split(".")[0]
            self.zip_file = list_of_names[0]
            with ZipFile(self.zip_file, "r") as file:
                    extract_folder = "img_extract_folder"
                    file.extractall(extract_folder)
            self.img_folder_path = os.path.join(extract_folder, folder_name)
    
class ImageClusterCreator(object):
    # def __init__(self, list_of_contents, list_of_names) -> None:
    #     self.list_of_contents = list_of_contents
    #     self.list_of_names = list_of_names
    #     folder_name = list_of_names[0].split(".")[0]
    #     self.zip_file = None
    #     with ZipFile(list_of_names[0], "r") as file:
    #             extract_folder = "img_extract_folder"
    #             file.extractall(extract_folder)
    #     self.img_folder_path = os.path.join(extract_folder, folder_name)
    @lru_cache(maxsize=None)   
    def extract_img_features(self, img_folder_path, method="pca", 
                             
                             ): 
        self.cl = Clustimage(method=method)
        self.extracted_feats = self.cl.fit_transform(img_folder_path)
        return self.extracted_feats
    
    def get_tsne_features(self, n_components=3, random_state=2023, **kwargs):
        self.features = self.extracted_feats['feat']
        #valid_transformed = cl.fit_transform(subset_valid_dir)

        #valid_result_feat = valid_transformed['feat']
        tsne_mod = TSNE(n_components=n_components, random_state=random_state, **kwargs)
        tsne_3feat = tsne_mod.fit_transform(self.features)
        self.extracted_feats['xyzcoord'] = tsne_3feat
        return self.extracted_feats
    
    def plot_3d_tsne_features(self, **kwargs):
        feature_results = self.get_tsne_features()
        fig = px.scatter_3d(feature_results['xyzcoord'], x=0, y=1, z=2,
                                color=feature_results['labels'], labels={'color': 'cluster'},
                                **kwargs
                            )
        fig.update_traces(marker_size=8)
        return fig         
            
    def plot_clustered_imgs(self, zoom=1, fig_height=150, fig_width=100,
                            plt_all=True, **kwargs):
        self.fig_clustered_imgs = self.cl.scatter(zoom=zoom, plt_all=plt_all, 
                        figsize=(fig_height,fig_width), **kwargs
                        )
        return self.fig_clustered_imgs
    @property    
    def img_cluster_result_df(self):
        results = self.cl.results
        results_selected = {key: value for key, value in results.items() 
                            if key not in ['img', 'feat', 'xycoord']
                            }
        self.results_cluster_df = pd.DataFrame.from_dict(results_selected).rename(columns={'labels': 'cluster'})
        return self.results_cluster_df
    
    def split_train_test_imgs(self):
        class DataSplitReturn(NamedTuple):
            train_df: pd.DataFrame
            test_df: pd.DataFrame
            
        results_cluster_df = self.img_cluster_result_df
        train_df, test_df = train_test_split(results_cluster_df, train_size=0.7, shuffle=True, random_state=2023,
                                    stratify=results_cluster_df[["cluster"]]
                                    )
        return DataSplitReturn(train_df=train_df, test_df=test_df)
        
    def plot_unique_imgs_per_cluster(self):
        fig_unique_img_per_cluster = self.cl.plot_unique(img_mean=False)
        return fig_unique_img_per_cluster
        
        
        
        