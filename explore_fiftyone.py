
#%%
import fiftyone as fo
from glob import glob
import json


img1 = "/home/lin/codebase/cv_with_roboflow_data/images/0.jpg"


    
#%%
sample = fo.Sample(filepath=img1)
# %%
sample["quality"] = 90
# %%
dataset = fo.Dataset(name="tomato_dataset")
dataset.add_sample(sample=sample)

#%%



#%%  load coco annotation into fiftyone
img_dir = "/home/lin/codebase/cv_with_roboflow_data/images/"
img_list = glob(f"{img_dir}*")[1:20]
coco_instance_anno = "/home/lin/codebase/cv_with_roboflow_data/tomato_coco_annotation/annotations/instances_default.json"

coco_dataset = fo.Dataset.from_dir(data_path=img_dir, dataset_type=fo.types.COCODetectionDataset,
                                  labels_path=coco_instance_anno,
                                  include_id=True
                                  )


#%%
session = fo.launch_app(coco_dataset)
#%%

coco_dataset.default_classes
#%%
with open(coco_instance_anno, "r") as filepath:
  coco_data = json.load(filepath)

#%%
categories = coco_data["category"]
category = {cat["id"]: cat["name"] for cat in categories}
for img in coco_data["images"]:
  img_name = img["file_name"]
  filepath = os.path.join(img_dir, img_name)
  sample = fo.Sample(filepath=filepath)
  img_id = img["id"]
  anns =[ann for ann in coco_data["annotations"] if ann["image_id"]==img_id]
  segmasks = [ann["segmentation"] for ann in anns]
  labels = [category[ann["id"]] for ann in anns]
  fo.Detection(instance=segmasks, label=labels)
    
  

#%%
samples = [fo.Sample(filepath=img) for img in img_list]

#%%

dataset_ld = fo.load_dataset(name="tomato_dataset")

#%%
dataset_ld.add_samples(samples=samples)

#%%
session = fo.launch_app(dataset=dataset_ld)


#%%
anno_key = "cvat_annot_example"

#%%
import os


print(os.getenv(key="FIFTYONE_CVAT_URL"))

#%%
passwd = os.environ["FIFTYONE_CVAT_PASSWORD"]
usrnm = os.environ["FIFTYONE_CVAT_USERNAME"]
url = os.environ["FIFTYONE_CVAT_URL"]
#%%

ann_res = dataset_ld.annotate(anno_key=anno_key, username=usrnm,
                    password=passwd, url=url,
                    launch_editor=True,
                    label_field="ground_truth",
                    label_type="instance",
                    classes=["ripe", "unripe", "flowers"]
                    )


#%%

ann_res

#%%

annotation_results = dataset_ld.load_annotation_results(anno_key=anno_key)


#%%
dataset_ld.load_annotations(anno_key=anno_key, username=usrnm,
                    password=passwd, url=url,
                    launch_editor=True,)
view = dataset_ld.load_annotation_view(anno_key=anno_key)
#%%
ann_app = fo.launch_app(view)
#%%

dataset_ld.delete_annotation_run(anno_key=anno_key)
# %%
sample.tags.append("train")
# %%
sample.tags
# %%
dataset.compute_metadata()
# %%
dataset.first()
# %%
sample["fruit"] = fo.Classification(label="tomato")
# %%
sample
# %%  creating view
import fiftyone.zoo as foz
import fiftyone.brain as fob
from fiftyone import ViewField as F

dataset = foz.load_zoo_dataset("cifar10", split="test")
cats = dataset.match(F("ground_truth.label")=="cat")
fob.compute_uniqueness(cats)
similar_cats = cats.sort_by("uniqueness", reverse=False)
session = fo.launch_app(view=similar_cats)

# %%
session.browser
# %%
session.close()
# %%  aggregations
quickstart_dataset = foz.load_zoo_dataset("quickstart")

#%%
print(quickstart_dataset.count_values("predictions.detections.label"))

#%%
quickstart_dataset.filter_labels("predictions", F("label")=="cat").bounds("predictions.detections.confidence")

#%%
fo.list_datasets()

#%%
# create a func that takes an image dir and coco_ann file and create a fiftyOne dataset obj
def create_dataset(img_dir, coco_annotation_file):
    pass

# %%
[
  {
    "name": "ripe",
    "color": "#e97c66",
    "type": "polygon",
    "attributes": []
  },
  {
    "name": "unripe",
    "color": "#64bbe8",
    "type": "polygon",
    "attributes": []
  },
  {
    "name": "flowers",
    "color": "#dff184",
    "type": "polygon",
    "attributes": []
  }
]



distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list