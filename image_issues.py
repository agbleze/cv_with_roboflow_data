

#%%
from cleanvision import Imagelab 
from random import sample

# %%
dataset_path = "/home/lin/codebase/cv_with_roboflow_data/field_crop_with_disease"
# %%

imagelab = Imagelab(data_path=dataset_path)

# %%
imagelab.find_issues()
# %%
imagelab.info
# %%
neardup_set = imagelab.info["near_duplicates"]["sets"]
# %%
for each_set in neardup_set:
    print(len(each_set))
    
    
    
# %%
imagelab.info["blurry"]
# %%
issues_found = imagelab.issue_summary[imagelab.issue_summary["num_images"]>0]["issue_type"].values.tolist()
# %%
for issue in issues_found:
    image_sets = imagelab.info[issue]["sets"]
    image_samples = [sample(img_set, 1) for img_set in image_sets]
