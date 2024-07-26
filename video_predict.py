
#%%
import cv2
import os
import torch
from torchvision.ops import nms
import glob
from torch_snippets import *
from PIL import Image
import json

#%%
prepath = path = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/preprocess_assets/"

target2label_json_path = prepath + "target2label.json"

with open(target2label_json_path, "r") as fp:
    target2label = json.loads(fp.read())
    

#%%
target2label = {int(k): v for k, v in target2label.items()}


#target2label.items   
    
#%%
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
def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                                              )
    img = normalize(img)
    return img.to(device).float()


#%%
full_model = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/model_save/full_model.pth"

img_path = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/tomato_fruit/"

#%%
img_list = sorted(glob.glob(f"""{img_path}*.jpg"""))
file_names = [os.path.basename(img) for img in img_list]

#%%

img = Image.open(img_list[0]).convert("RGB").resize((640, 640))
img.size

#%%
model = torch.load(full_model)
model.eval()
#%%
for file_name, img in zip(file_names, img_list):  
    img = Image.open(img).convert("RGB").resize((640, 640))      
    img = np.array(img)/255
    show(img)
    prep_img = preprocess_image(img=img)
    model_res = model([prep_img])
    #print(file_name)
    #print(model_res)
    #break
    bbs, confs, labels = decode_output(model_res[0])
    info = [f'{l}@{c:.2f}' for l,c in zip(labels, confs)]
    print(file_name)
    print(info)
    print(labels)
    if len(bbs) != 0:
        show(img, bbs=bbs, texts=labels, sz=5, text_sz=10,
            title=info)
    else:
        print("No bbs")



#%%
video_img_path = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/images/"
video_img_list = sorted(glob.glob(f"""{video_img_path}*.jpg"""))
video_img_names = [os.path.basename(img) for img in img_list]


img_cv1 = cv2.imread(video_img_list[15])#.resize((640, 640))

img_resize = cv2.resize(img_cv1, (640,640))
img_rgbcolor = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
show(img_rgbcolor)

#%%
video_imgarray = np.array(img_rgbcolor)/255
prep_video_img = preprocess_image(img=video_imgarray)

#%%
prep_video_img.shape

#%%
model_res = model([prep_video_img])
bbs, confs, labels = decode_output(model_res[0])

bbs
#%%
#x1, y1, x2, y2 = bbs

#%%
for field_img in video_img_list:
    field_img = Image.open(field_img).convert("RGB").resize((640,640))
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
    if len(bbs) != 0:
        show(field_img, bbs=bbs, texts=labels, sz=5, text_sz=10,
            title=info)
    
#%%

#model()


#%%
#video_capture = cv2.VideoCapture(0)

video_file_path = "C:/Users/agbji/Documents/codebase/image_extraction/field_video.mp4"

video_file = cv2.VideoCapture(video_file_path)

#%%
x_shape = int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
y_shape = int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))

four_cc =cv2.VideoWriter_fourcc(*"MJPG")

#%%
vid_outfile = cv2.VideoWriter("only_disease_pred_video.avi", four_cc, 20, (x_shape, y_shape))



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
            #cv2.putText(frame, label)
            
            
        
    ## draw predicted bbox
    
    for (x1, y1, x2, y2) in bbox:
        cv2.rectangle(img=frame,
                      pt1=(x1, y1),
                      pt2=(x2, y2)
                      color=(0, 255, 0),
                      thickness=0
                      )
        cv2.putText(frame, label)
        
    cv2.imshow("Crop Disease prediction", frame)
    
    if cv2.waitKey(1) == ord("q"):
        break
    
video_capture.release()
cv2.destroyAllWindows()
    
    
# %%
