

#%%
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.algorithms.hash_key_inference.prune import Prune

img_dir = "/home/lin/codebase/__cv_with_roboflow_data/field_crop_with_disease"

env = Environment()
detected_format = env.detect_dataset(path=img_dir)

# %%

dataset = Dataset.import_from(img_dir, detected_format[0])
prune = Prune(dataset, cluster_method="random")


#%%
data = prune.get_pruned()

# %%
data.infos()
# %%

import datumaro as dm
from datumaro.components.algorithms.hash_key_inference.prune import Prune
# %%
dataset = dm.Dataset.import_from("caltech-101", format="imagenet")
dataset
# %% generate the validation report
from datumaro.plugins.validators import ClassificationValidator
from matplotlib import pyplot as plt
# %%
validator = ClassificationValidator()
reports = validator.validate(dataset)
# %%
stats = reports["statistics"]
# %%
label_stats = stats["label_distribution"]["defined_labels"]
label_name, label_counts = zip(*[(k,v) for k, v in label_stats.items()])

plt.figure(figsize=(12,4))
plt.hist(label_name, weights=label_counts, bins=len(label_name))
plt.xticks(rotation="vertical")
plt.show()

#%%
prune = Prune(dataset, cluster_method="random")
random_result = prune.get_pruned(0.5)

random_reports = validator.validate(random_result)
random_stats = random_reports["statistics"]

label_stats = random_stats["label_distribution"]["defined_labels"]
label_name, label_counts = zip(*[(k, v) for k, v in label_stats.items()])

plt.figure(figsize=(12, 4))
plt.hist(label_name, weights=label_counts, bins=len(label_name))
plt.xticks(rotation="vertical")
plt.show()


#%%  ###  cluster_random  ###
prune = Prune(dataset, cluster_method="cluster_random")
cluster_random_result = prune.get_pruned(0.5)

#%%
cluster_random_reports = validator.validate(cluster_random_result)

cluster_random_stats = cluster_random_reports["statistics"]

label_stats = cluster_random_stats["label_distribution"]["defined_labels"]
label_name, label_counts = zip(*[(k, v) for k, v in label_stats.items()])

plt.figure(figsize=(12, 4))
plt.hist(label_name, weights=label_counts, bins=len(label_name))
plt.xticks(rotation="vertical")
plt.show()

#%%  ####  use query_clust method  ###
## query clust method needs to be debugged for valueerror
prune = Prune(dataset, cluster_method="query_clust")
quuery_clust_result = prune.get_pruned(0.3)

#%%  ####  centroid  ####
prune = Prune(dataset, cluster_method="centroid")
centroid_result = prune.get_pruned(0.5)

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
entropy_result = prune.get_pruned(0.5)

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

 

