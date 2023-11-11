# %% [markdown]
# <a href="https://colab.research.google.com/github/PacktPublishing/Hands-On-Computer-Vision-with-PyTorch/blob/master/Chapter08/Training_Faster_RCNN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
import os
if not os.path.exists('images'):
    !pip install -qU torch_snippets
    from google.colab import files
    files.upload() # upload kaggle.json
    !mkdir -p ~/.kaggle
    !mv kaggle.json ~/.kaggle/
    !ls ~/.kaggle
    !chmod 600 /root/.kaggle/kaggle.json
    !kaggle datasets download -d sixhky/open-images-bus-trucks/
    !unzip -qq open-images-bus-trucks.zip
    !rm open-images-bus-trucks.zip

# %%
import os
from torch_snippets import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
from sklearn.model_selection import train_test_split
import pandas as pd
import torchvision.transforms as transforms
from cluster_utils import coco_annotation_to_df


#%%
annot_path = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/train/_annotations.coco.json"

annot_df = coco_annotation_to_df(coco_annotation_file=annot_path)


#%%
IMAGE_ROOT = '/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/train/'#'images/images'
#df = '/Users/lin/Documents/python_venvs/modern_cv_with_pytorch/chap8_advanced_obj_detection/open-images-bus-trucks/df.csv'
#DF_RAW = pd.read_csv(df)


#%%

annot_df.info()


#%%



#%%
sorted_img_list = sorted(glob.glob(f"""{IMAGE_ROOT}*.jpg"""))[:100]


#%%
annot_df.file_name
#some_img = sorted_img_list[:100]

#%%
file_names = [img.split('/')[-1] for img in sorted_img_list]

#%%

subset_df = annot_df[annot_df['file_name'].isin(file_names)]

#%%
img1_path =find(file_names[0], file_names)

#%%


img = Image.open(sorted_img_list[0]).convert("RGB")

#%%
import matplotlib.pyplot as plt

#%%
img_array = plt.imread(sorted_img_list[0])


#%%

img1_bbs = subset_df[subset_df['file_name']==file_names[0]]['bbox'].tolist()

#%%

img1_labels = subset_df[subset_df['file_name']==file_names[0]]['category_name'].tolist()


#%%

show(img_array, bbs=img1_bbs) 


#%%
img1_bbs[0][2] + img1_bbs[0][0]



#%%
img1_bbs_trns = [[imgbbx[0], imgbbx[1], imgbbx[0]+imgbbx[2], imgbbx[1]+imgbbx[3]] 
                 for imgbbx in img1_bbs]

#%%
img1_bbs_trns

#%%

subset_bbx_df = subset_df.copy()


    
#%% create bbox xmin, ymin, xmax, ymax

subset_bbx_df[['XMin', 'YMin', 'width', 'height']] =  pd.DataFrame(subset_bbx_df.bbox.tolist(), index= subset_bbx_df.index)

subset_bbx_df['XMax'] = subset_bbx_df['XMin'] + subset_bbx_df['width']

subset_bbx_df['YMax'] = subset_bbx_df['YMin'] + subset_bbx_df['height']

#%%
show(img_array, bbs=img1_bbs_trns, texts=img1_labels, text_sz=15)

#%%
import shutil

#%%
subset_img_path = "/Users/lin/Documents/python_venvs/modern_cv_with_pytorch/chap8_advanced_obj_detection/subset_img/"

#[shutil.copy(img_path, subset_img_path) for img_path in some_img]


# %%
"""from torch_snippets import *
from PIL import Image
IMAGE_ROOT = 'images/images'
DF_RAW = df = pd.read_csv('df.csv')"""


#%%

label2target = {cat_nm: cat_id for cat_nm, cat_id in 
 zip(subset_df.category_name.unique(), subset_df.category_id.unique())
 }

label2target['background'] = 0
target2label = {t:l for l,t in label2target.items()}
background_class = label2target['background']
num_classes = len(label2target)

# %%
label2target = {l:t+1 for t,l in enumerate(subset_bbx_df['category_name'].unique())}
label2target['background'] = 0
target2label = {t:l for l,t in label2target.items()}
background_class = label2target['background']
num_classes = len(label2target)

# %%
def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    return img.to(device).float()

# %%
class OpenDataset(torch.utils.data.Dataset):
    w, h = 224, 224
    def __init__(self, df, image_dir=IMAGE_ROOT):
        self.image_dir = image_dir
        self.files = glob.glob(self.image_dir+'/*')
        self.df = df
        self.image_infos = df.file_name.unique()
    def __getitem__(self, ix):
        # load images and masks
        image_id = self.image_infos[ix]
        img_path = find(image_id, self.files)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR))/255.
        data = self.df[self.df['file_name'] == image_id]
        labels = data['category_name'].values.tolist()
        data = data[['XMin','YMin','XMax','YMax']].values
        data[:,[0,2]] *= self.w
        data[:,[1,3]] *= self.h
        boxes = data.astype(np.uint32).tolist() # convert to absolute coordinates
        # torch FRCNN expects ground truths as a dictionary of tensors
        target = {}
        target["boxes"] = torch.Tensor(boxes).float()
        target["labels"] = torch.Tensor([label2target[i] for i in labels]).long()
        img = preprocess_image(img)
        return img, target
    def collate_fn(self, batch):
        return tuple(zip(*batch)) 

    def __len__(self):
        return len(self.image_infos)

# %%
from sklearn.model_selection import train_test_split
trn_ids, val_ids = train_test_split(subset_bbx_df.file_name.unique(), test_size=0.1, random_state=99)
trn_df, val_df = subset_bbx_df[subset_bbx_df['file_name'].isin(trn_ids)], subset_bbx_df[subset_bbx_df['file_name'].isin(val_ids)]
len(trn_df), len(val_df)

train_ds = OpenDataset(trn_df)
test_ds = OpenDataset(val_df)

train_loader = DataLoader(train_ds, batch_size=4, collate_fn=train_ds.collate_fn, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=4, collate_fn=test_ds.collate_fn, drop_last=True)

# %%
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# %%
# Defining training and validation functions for a single batch
def train_batch(inputs, model, optimizer):
    model.train()
    input, targets = inputs
    input = list(image.to(device) for image in input)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    optimizer.zero_grad()
    losses = model(input, targets)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optimizer.step()
    return loss, losses

@torch.no_grad() # this will disable gradient computation in the function below
def validate_batch(inputs, model):
    model.train() # to obtain the losses, model needs to be in train mode only. # #Note that here we are not defining the model's forward method 
#and hence need to work per the way the model class is defined
    input, targets = inputs
    input = list(image.to(device) for image in input)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    optimizer.zero_grad()
    losses = model(input, targets)
    loss = sum(loss for loss in losses.values())
    return loss, losses

# %%
model = get_model().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
n_epochs = 5
log = Report(n_epochs)

# %%
for epoch in range(n_epochs):
    _n = len(train_loader)
    for ix, inputs in enumerate(train_loader):
        loss, losses = train_batch(inputs, model, optimizer)
        loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = \
            [losses[k] for k in ['loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg']]
        pos = (epoch + (ix+1)/_n)
        log.record(pos, trn_loss=loss.item(), trn_loc_loss=loc_loss.item(), 
                   trn_regr_loss=regr_loss.item(), trn_objectness_loss=loss_objectness.item(),
                   trn_rpn_box_reg_loss=loss_rpn_box_reg.item(), end='\r')

    _n = len(test_loader)
    for ix,inputs in enumerate(test_loader):
        loss, losses = validate_batch(inputs, model)
        loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = \
          [losses[k] for k in ['loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg']]
        pos = (epoch + (ix+1)/_n)
        log.record(pos, val_loss=loss.item(), val_loc_loss=loc_loss.item(), 
                  val_regr_loss=regr_loss.item(), val_objectness_loss=loss_objectness.item(),
                  val_rpn_box_reg_loss=loss_rpn_box_reg.item(), end='\r')
    if (epoch+1)%(n_epochs//5)==0: log.report_avgs(epoch+1)


#%%

#Report.plot_epochs()

# %%
log.plot_epochs(['trn_loss'])

# %%
from torchvision.ops import nms
def decode_output(output):
    'convert tensors to numpy arrays'
    bbs = output['boxes'].cpu().detach().numpy().astype(np.uint16)
    labels = np.array([target2label[i] for i in output['labels'].cpu().detach().numpy()])
    confs = output['scores'].cpu().detach().numpy()
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

    if len(ixs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()

# %%
model.eval()

#%%
for ix, (images, targets) in enumerate(test_loader):
    if ix==5: break
    images = [im for im in images]
    #show(images[0])
    outputs = model(images)
    for ix, output in enumerate(outputs):
        bbs, confs, labels = decode_output(output)
        info = [f'{l}@{c:.2f}' for l,c in zip(labels, confs)]
        show(images[ix].cpu().permute(1,2,0), bbs=bbs, texts=labels, sz=5, text_sz=10,
             title=info)

# %%





# %%
