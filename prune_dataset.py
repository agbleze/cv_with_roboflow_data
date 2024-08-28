

#%%
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.algorithms.hash_key_inference.prune import Prune
from datumaro.plugins.validators import DetectionValidator, SegmentationValidator
img_dir = "/home/lin/codebase/instance_segmentation/dataset_to_sample"
env = Environment()
detected_format = env.detect_dataset(path=img_dir)

# %%

dataset = Dataset.import_from(img_dir, detected_format[0])

#%%  ###  cluster_random  ###
prune = Prune(dataset, cluster_method="cluster_random")
cluster_random_result = prune.get_pruned(0.5)

#%%


#%%
validator = SegmentationValidator()
cluster_random_reports = validator.validate(cluster_random_result)

cluster_random_stats = cluster_random_reports["statistics"]

label_stats = cluster_random_stats["label_distribution"]["defined_labels"]
label_name, label_counts = zip(*[(k, v) for k, v in label_stats.items()])

plt.figure(figsize=(12, 4))
plt.hist(label_name, weights=label_counts, bins=len(label_name))
plt.xticks(rotation="vertical")
plt.show()


#%%
repsave = "/home/lin/codebase/instance_segmentation/cocoa-ripeness-inst.v2i.coco-segmentation/repsample_cluster_random"
cluster_random_result.export(repsave, format="coco_instances", save_media=True)


#%%  ####  use query_clust method  ###
## query clust method needs to be debugged for valueerror
#prune = Prune(dataset, cluster_method="query_clust")
#quuery_clust_result = prune.get_pruned(0.3)

#%%  ####  centroid  ####
prune = Prune(dataset, cluster_method="centroid")
centroid_result = prune.get_pruned(0.5)

#%%
repsave = "/home/lin/codebase/instance_segmentation/repsample_centroid"
centroid_result.export(repsave, format="coco_instances", save_media=True)


#%%
"""
Epoch 19/19 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 270/270 0:02:05 • 0:00:00 2.18it/s v_num: 0 train/loss_rpn_cls: 0.000              
                                                                                        train/loss_rpn_bbox: 0.002 train/loss_cls: 0.015
                                                                                        train/loss_bbox: 0.020 train/loss_mask: 0.037   
                                                                                        train/loss: 0.074 validation/data_time: 0.005   
                                                                                        validation/iter_time: 0.083 val/map: 0.667      
                                                                                        val/map_50: 0.730 val/map_75: 0.720             
                                                                                        val/map_small: 0.201 val/map_medium: 0.534      
                                                                                        val/map_large: 0.711 val/mar_1: 0.652           
                                                                                        val/mar_10: 0.897 val/mar_100: 0.897            
                                                                                        val/mar_small: 0.322 val/mar_medium: 0.860      
                                                                                        val/mar_large: 0.953 val/map_per_class: -1.000  
                                                                                        val/mar_100_per_class: -1.000 val/f1-score:     
                                                                                        0.711 train/data_time: 0.012 train/iter_time:   
                                                                                        0.461                                           
Elapsed time: 0:56:42.451325

"""

## adjustment needs to be made for pruning to take and prune on train and val
# keep test separate as it is currently assummed that all the dataset is 
# being combined before pruning hence may not be comparable in terms of 
# getting a test dataset

#%%
centroid_reports = validator.validate(centroid_result)

centroid_stats = centroid_reports["statistics"]

label_stats = centroid_stats["label_distribution"]["defined_labels"]
label_name, label_counts = zip(*[(k, v) for k, v in label_stats.items()])

plt.figure(figsize=(12, 4))
plt.hist(label_name, weights=label_counts, bins=len(label_name))
plt.xticks(rotation="vertical")
plt.show()

#%%  ### Entropy  ###
prune = Prune(dataset, cluster_method="entropy")

#%%
entropy_result = prune.get_pruned(0.9)

#%%
entropy_reports = validator.validate(entropy_result)

entropy_stats = entropy_reports["statistics"]

label_stats = entropy_stats["label_distribution"]["defined_labels"]
label_name, label_counts = zip(*[(k, v) for k, v in label_stats.items()])

plt.figure(figsize=(12, 4))
plt.hist(label_name, weights=label_counts, bins=len(label_name))
plt.xticks(rotation="vertical")
plt.show()


# %%  ### Near duplicate removal
prune = Prune(dataset, cluster_method="ndr")
ndr_result = prune.get_pruned(0.5)

#%%
repsave = "/home/lin/codebase/instance_segmentation/repsample_ndr"

ndr_result.export(repsave, format="coco_instances", save_media=True)




# %%
ndr_reports = validator.validate(ndr_result)

ndr_stats = ndr_reports["statistics"]

label_stats = ndr_stats["label_distribution"]["defined_labels"]
label_name, label_counts = zip(*[(k, v) for k, v in label_stats.items()])

plt.figure(figsize=(12, 4))
plt.hist(label_name, weights=label_counts, bins=len(label_name))
plt.xticks(rotation="vertical")
plt.show()
# %%
random_result.export("random_result", format="datumaro", save_media=True)
cluster_random_result.export("cluster_random_result", format="datumaro", save_media=True)
#query_clust_result.export("query_clust_result", format="datumaro", save_media=True)
centroid_result.export("centroid_result", format="datumaro", save_media=True)
entropy_result.export("entropy_result", format="datumaro", save_media=True)
ndr_result.export("ndr_result", format="datumaro", save_media=True)

#%%
random_result.export("random_result_imagenetFormat", format="imagenet_with_subset_dirs", save_media=True)

# %%   ###train model with full data

# %%  experiment to determine if representative sampling reduces training time while 
# minimally compromising on the accuracy at an acceptable level

 
#%%

datapath = "/home/lin/codebase/instance_segmentation/cocoa-ripeness-inst.v2i.coco-segmentation"
dm.Dataset.from_import(datapath)

# %%
from glob import glob

val_dir = "/home/lin/codebase/instance_segmentation/cocoa-ripeness-inst.v2i.coco-segmentation/repsample_centroid/images/val"

len(glob(f"{val_dir}/*"))
# %%
"""
ALL DATA TRAINING RESULT

Epoch 16/19 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 603/603 0:04:21 • 0:00:00 2.23it/s v_num: 0 train/loss_rpn_cls: 0.002               
                                                                                        train/loss_rpn_bbox: 0.002 train/loss_cls: 0.011 
                                                                                        train/loss_bbox: 0.025 train/loss_mask: 0.040    
                                                                                        train/loss: 0.079 validation/data_time: 0.005    
                                                                                        validation/iter_time: 0.087 val/map: 0.720       
                                                                                        val/map_50: 0.778 val/map_75: 0.771              
                                                                                        val/map_small: 0.198 val/map_medium: 0.565       
                                                                                        val/map_large: 0.775 val/mar_1: 0.641 val/mar_10:
                                                                                        0.910 val/mar_100: 0.910 val/mar_small: 0.549    
                                                                                        val/mar_medium: 0.882 val/mar_large: 0.960       
                                                                                        val/map_per_class: -1.000 val/mar_100_per_class: 
                                                                                        -1.000 val/f1-score: 0.683 train/data_time: 0.013
                                                                                        train/iter_time: 0.433                           
Elapsed time: 1:40:05.112484
"""


#%%


"""
# cluster random representative sampling results

Epoch 19/19 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 307/307 0:02:16 • 0:00:00 2.21it/s v_num: 0 train/loss_rpn_cls: 0.002 train/loss_rpn_bbox: 
                                                                                        0.002 train/loss_cls: 0.006 train/loss_bbox: 0.009      
                                                                                        train/loss_mask: 0.054 train/loss: 0.073                
                                                                                        validation/data_time: 0.005 validation/iter_time: 0.103 
                                                                                        val/map: 0.739 val/map_50: 0.804 val/map_75: 0.792      
                                                                                        val/map_small: 0.235 val/map_medium: 0.602              
                                                                                        val/map_large: 0.795 val/mar_1: 0.657 val/mar_10: 0.904 
                                                                                        val/mar_100: 0.905 val/mar_small: 0.617 val/mar_medium: 
                                                                                        0.881 val/mar_large: 0.951 val/map_per_class: -1.000    
                                                                                        val/mar_100_per_class: -1.000 val/f1-score: 0.729       
                                                                                        train/data_time: 0.013 train/iter_time: 0.442           
Elapsed time: 0:58:05.186021

"""