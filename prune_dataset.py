

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
prune.get_pruned()

# %%
