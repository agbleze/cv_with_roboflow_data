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
from torchvision import transforms
import json


#%%
annot_path = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/Tomato-pest&diseases-1/train/_annotations.coco.json"  #"/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/train/_annotations.coco.json"

annot_df = coco_annotation_to_df(coco_annotation_file=annot_path).dropna()


#%%
TRAIN_IMG_ROOT = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/Tomato-pest&diseases-1/train/" #'/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/train/'#'images/images'
#df = '/Users/lin/Documents/python_venvs/modern_cv_with_pytorch/chap8_advanced_obj_detection/open-images-bus-trucks/df.csv'
#DF_RAW = pd.read_csv(df)

VAL_IMG_ROOT = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/Tomato-pest&diseases-1/valid/"

INFER_IMG_ROOT = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/Tomato-pest&diseases-1/test/"
#%%

annot_df.info()


#%%



#%%
sorted_img_list = sorted(glob.glob(f"""{TRAIN_IMG_ROOT}*.jpg"""))#[:100]


#%%
annot_df.file_name
#some_img = sorted_img_list[:100]

#%%  ### changed \ to \\ becuase started working on windows
file_names = [os.path.basename(img) for img in sorted_img_list]


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

img_array.shape
#%%

img1_bbs = subset_df[subset_df['file_name']==file_names[0]]['bbox'].tolist()

#%%

img1_labels = subset_df[subset_df['file_name']==file_names[0]]['category_name'].tolist()


#%%

show(img_array, bbs=img1_bbs) 


#%%
#img1_bbs[0][2] + img1_bbs[0][0]



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

def transform_data_bbox(coco_annot_filepath):
    annot_df = coco_annotation_to_df(coco_annotation_file=coco_annot_filepath).dropna()
    annot_df[['XMin', 'YMin', 'width', 'height']] =  pd.DataFrame(annot_df.bbox.tolist(), index= annot_df.index)

    annot_df['XMax'] = annot_df['XMin'] + annot_df['width']

    annot_df['YMax'] = annot_df['YMin'] + annot_df['height']
    return annot_df


#%%
train_coco_file = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/Tomato-pest&diseases-1/train/_annotations.coco.json"

val_coco_file = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/Tomato-pest&diseases-1/valid/_annotations.coco.json"
inference_coco_file = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/Tomato-pest&diseases-1/test/_annotations.coco.json"
train_df = transform_data_bbox(coco_annot_filepath=train_coco_file)
val_df = transform_data_bbox(val_coco_file)
infer_df = transform_data_bbox(coco_annot_filepath=inference_coco_file)
#%%
show(img_array, bbs=img1_bbs_trns, texts=img1_labels, text_sz=15)

#%%
import shutil

#%%
#subset_img_path = "/Users/lin/Documents/python_venvs/modern_cv_with_pytorch/chap8_advanced_obj_detection/subset_img/"

#[shutil.copy(img_path, subset_img_path) for img_path in some_img]


# %%
"""from torch_snippets import *
from PIL import Image
IMAGE_ROOT = 'images/images'
DF_RAW = df = pd.read_csv('df.csv')"""


#%%

#label2target = {cat_nm: cat_id for cat_nm, cat_id in 
# zip(subset_df.category_name.unique(), subset_df.category_id.unique())
# }

#label2target['background'] = 0
#target2label = {t:l for l,t in label2target.items()}
#background_class = label2target['background']
#num_classes = len(label2target)

# %%
label2target = {l:t+1 for t,l in enumerate(subset_bbx_df['category_name'].unique())}
label2target['background'] = 0
target2label = {int(t):l for l,t in label2target.items()}
background_class = label2target['background']
num_classes = len(label2target)


#%%
prepath = path = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/preprocess_assets/"

target2label_json_path = prepath + "target2label.json"

with open(target2label_json_path, "w") as fp:
    json.dump(target2label, fp)

#%%
label2target_json_path = prepath + "label2target.json"

with open(label2target_json_path, "w") as fp:
    json.dump(label2target, fp)

#%%
"""with open(target2label_json_path, "r") as fp:
    target2label = json.loads(fp.read())
"""

# %%
def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                                              )
    img = normalize(img)
    return img.to(device).float()

#%%

#preprocess_image(img=img_array)
# %%
class OpenDataset(torch.utils.data.Dataset):
    w, h = 224, 224
    def __init__(self, df, image_dir=TRAIN_IMG_ROOT):
        self.image_dir = image_dir
        self.files = glob.glob(self.image_dir+'/*')
        self.df = df
        self.image_infos = df.file_name.unique()
    def __getitem__(self, ix):
        # load images and masks
        image_id = self.image_infos[ix]
        img_path = find(image_id, self.files)
        img = Image.open(img_path).convert("RGB")
        #img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR))/255.
        img = np.array(img)/255
        data = self.df[self.df['file_name'] == image_id]
        labels = data['category_name'].values.tolist()
        data = data[['XMin','YMin','XMax','YMax']].values
        #data[:,[0,2]] *= self.w
        #data[:,[1,3]] *= self.h
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

#%%



# %%
#from sklearn.model_selection import train_test_split
#trn_ids, val_ids = train_test_split(subset_bbx_df.file_name.unique(), test_size=0.1, random_state=99)
#trn_df, val_df = subset_bbx_df[subset_bbx_df['file_name'].isin(trn_ids)], subset_bbx_df[subset_bbx_df['file_name'].isin(val_ids)]
#len(trn_df), len(val_df)

#train_ds = OpenDataset(trn_df)
#test_ds = OpenDataset(val_df)


#%%
train_ds = OpenDataset(train_df, image_dir=TRAIN_IMG_ROOT)
test_ds = OpenDataset(val_df, image_dir=VAL_IMG_ROOT)
infer_ds = OpenDataset(infer_df, image_dir=INFER_IMG_ROOT)

train_loader = DataLoader(train_ds, batch_size=4, collate_fn=train_ds.collate_fn, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=4, collate_fn=test_ds.collate_fn, drop_last=True)
infer_loader = DataLoader(infer_ds, batch_size=4, collate_fn=infer_ds.collate_fn, drop_last=True)
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
n_epochs = 300
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
    if (epoch+1)%(n_epochs//300)==0: log.report_avgs(epoch+1)


#%%

#Report.plot_epochs()

# %%
log.plot_epochs(['val_loss', 'trn_loss'])

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



#%%

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
    
    
#%%
import os 
model_save_path = os.path.join("C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/model_save", "model.pth") 
#torch.save(model.state_dict(), )


#%%  save model at checkpoint - saves the model state
torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, model_save_path
           )
    
    
#%% save entire model
full_model_save_path = os.path.join("C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/model_save", "full_model.pth") 

#torch.save(model, full_model_save_path)


#%%

model = torch.load(full_model_save_path)

# %%
model.eval()

#%%
pred_ds = OpenDataset(infer_df, image_dir=INFER_IMG_ROOT)
pred_loader = DataLoader(pred_ds, batch_size=1, collate_fn=infer_ds.collate_fn, drop_last=True)

for ix, (images, targets) in enumerate(pred_loader):
    if ix==5: break
    images = [im for im in images]
    #show(images[0])
    #print(len(images))
    outputs = model(images)
    for ix, output in enumerate(outputs):
        bbs, confs, labels = decode_output(output)
        print(bbs)
        print
        info = [f'{l}@{c:.2f}' for l,c in zip(labels, confs)]
        if len(bbs) != 0:
            show(images[ix].cpu().permute(1,2,0), bbs=bbs, texts=labels, sz=5, text_sz=10,
              title=info)


#%%
for ix, (images, targets) in enumerate(pred_loader):
    if ix==1: break
    images = [im for im in images]
    show(images[0])


# %%
img_list = sorted(glob.glob(f"""{INFER_IMG_ROOT}*.jpg"""))
infer_file_names = [os.path.basename(img) for img in img_list]
#img = Image.open(img_list[0]).convert("RGB")
#img = Image.open(img_path).convert("RGB")
        #img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR))/255.
        
for file_name, img in zip(infer_file_names, img_list):  
    img = Image.open(img).convert("RGB")      
    img = np.array(img)/255
    #show(img)
    prep_img = preprocess_image(img=img)
    model_res = model([prep_img])
    print(file_name)
    #print(model_res)
    #break
    bbs, confs, labels = decode_output(model_res[0])
    info = [f'{l}@{c:.2f}' for l,c in zip(labels, confs)]
    print(file_name)
    if len(bbs) != 0:
        show(img, bbs=bbs, texts=labels, sz=5, text_sz=10,
            title=info)


#%% prediction of field data
import shutil
dest_dir = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/field_crop_with_disease/"
field_path = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/images/"

#field_path = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/tomato_fruit/"
field_img_list = sorted(glob.glob(f"{field_path}*.jpg"))

#%%
field_img = Image.open(field_img_list[0]).convert("RGB")#.resize((640,640))

w, h = field_img.size

#%%
for field_img in field_img_list:
    field_img_read = Image.open(field_img).convert("RGB").resize((640,640))
    field_img_arr = np.array(field_img_read)
    field_img_array = np.array(field_img_read)/255
        #show(img)
    prep_img = preprocess_image(img=field_img_array)
    model_res = model([prep_img])
    #print(file_name)
    #print(model_res)
    #break
    bbs, confs, labels = decode_output(model_res[0])
    info = [f'{l}@{c:.2f}' for l,c in zip(labels, confs)]
    #print(file_name)
 
    if len(bbs) != 0:
        img_pred = []
        img_pred_i = show(field_img_read, bbs=bbs, texts=labels, sz=5, text_sz=10,
            title=info)
        img_pred.append(img_pred_i)
        shutil.move(field_img, dest_dir)
    
    """
    window_name = 'Image'
    font = cv2.FONT_HERSHEY_SIMPLEX 
    #org = (50, 50) 
    fontScale = 0.6
    color = (255, 0, 0) 
    thickness = 1 
    if len(bbs) == 0:
        x1,y1, x2, y2 = 0, 0, 0, 0
        frame = cv2.rectangle(img=field_img_arr,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      color=(0, 255, 0),
                      thickness=0
                      )
        #frame = cv2.putText(frame, label)
        
        frame = cv2.putText(img=field_img_arr, text="No prediction", 
                        org= (50, 50), 
                            fontFace= font,  
                            fontScale=fontScale, 
                            color=color, 
                            thickness=thickness, 
                            lineType=cv2.LINE_AA
                            )
        print(type(frame))
        show(frame)
        #cv2.imshow("test", frame)
    """
#%%
def predict(model, bytes_img):
    
    window_name = 'Image'
    font = cv2.FONT_HERSHEY_SIMPLEX 
    #org = (50, 50) 
    fontScale = 0.6
    color = (255, 0, 0) 
    thickness = 1 
            
            
    field_img = Image.open(bytes_img).convert("RGB").resize((640,640))
    field_img = np.array(field_img)/255
        #show(img)
    prep_img = preprocess_image(img=field_img)
    model_res = model([prep_img])
    #print(file_name)
    #print(model_res)
    #break
    bbs, confs, labels = decode_output(model_res[0])
    info = [f'{l}@{c:.2f}' for l,c in zip(labels, confs)]
    #print(file_name)
    if len(bbs) == 0:
        x1,y1, x2, y2 = 0, 0, 0, 0
        frame = cv2.rectangle(img=frame,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      color=(0, 255, 0),
                      thickness=0
                      )
        frame = cv2.putText(frame, label)
        
        frame = cv2.putText(img=frame, text="No prediction", 
                        org= (50, 50), 
                            fontFace= font,  
                            fontScale=fontScale, 
                            color=color, 
                            thickness=thickness, 
                            lineType=cv2.LINE_AA
                            )
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    else:
        #show(frame, bbs=bbs, texts=labels, sz=5, text_sz=10,
        #    title=info)
        #break
        #print("in else:")
        for i in range(n):
            print(f"bbs: {bbs}")
            x1, y1, x2, y2 = bbs[i]
            #x1, y1 = x1 * x_shape, y1 * y_shape
            #x2, y2 = x2 * x_shape, y2 * y_shape
            
            label = labels[i]
            bgr = (0,0, 255)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
            
               
            frame = cv2.putText(img=frame, text=label, 
                        org= (x1, y1), 
                            fontFace= font,  
                            fontScale=fontScale, 
                            color=color, 
                            thickness=thickness, 
                            lineType=cv2.LINE_AA
                            )
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            
            #print(label)
            #print(x1,y1, x2,y2)
    
    
    cv2.imshow("diseasd pred", frame)
    vid_outfile.write(frame)    

#%%

 ## load model
from time import time 

assert video_file.isOpened() 
while True:
    ret, frame = video_file.read()
    if not ret:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #show(frame)
    frame_norm_array = np.array(frame)/255
    #Image.open(field_img).convert("RGB").resize((640,640))
        #show(img)
    prep_frame = preprocess_image(img=frame_norm_array)
    model_res = model([prep_frame])
    
    bbs, confs, labels = decode_output(model_res[0])
    #print(bbs, labels)
    n = len(labels)
    info = [f'{l}@{c:.2f}' for l,c in zip(labels, confs)]
    #print(file_name)
    if len(bbs) == 0:
        x1,y1, x2, y2 = 0, 0, 0, 0
        frame = cv2.rectangle(img=frame,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      color=(0, 255, 0),
                      thickness=0
                      )
        #cv2.putText(frame, label)
    else:
        #show(frame, bbs=bbs, texts=labels, sz=5, text_sz=10,
        #    title=info)
        #break
        #print("in else:")
        for i in range(n):
            print(f"bbs: {bbs}")
            x1, y1, x2, y2 = bbs[i]
            #x1, y1 = x1 * x_shape, y1 * y_shape
            #x2, y2 = x2 * x_shape, y2 * y_shape
            
            label = labels[i]
            bgr = (0,0, 255)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
            
            window_name = 'Image'
            font = cv2.FONT_HERSHEY_SIMPLEX 
            #org = (50, 50) 
            fontScale = 0.6
            color = (255, 0, 0) 
            thickness = 1    
            frame = cv2.putText(img=frame, text=label, 
                        org= (x1, y1), 
                            fontFace= font,  
                            fontScale=fontScale, 
                            color=color, 
                            thickness=thickness, 
                            lineType=cv2.LINE_AA
                            )
            print(label)
            print(x1,y1, x2,y2)
    cv2.imshow("diseasd pred", frame)
    vid_outfile.write(frame)
        
    if cv2.waitKey(1) == ord("q"):
        break
    
video_file.release()
vid_outfile.release()
cv2.destroyAllWindows()

        
#%%    
img_pred    
    
#%%
show(prep_img)
#%%
prep_img.shape

