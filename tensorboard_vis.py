

#%%
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2
 
import requests
from zipfile import ZipFile
import os
import tensorflow as tf
from PIL import Image
 
from tensorboard.plugins import projector
from tensorboard import notebook
from clusteval import clusteval
from cluster_utils import (crop_image_with_bbox, coco_annotation_to_df, 
                           ImgPropertySetReturnType, get_img_names_labels_paths
                           )

#%%
seed = 2023
np.random.seed(seed)
# num_points_per_class = 50

# #%%
# # Class 1
# mean1 = [0, 0]
# cov = [[0.1, 0], [0, 0.1]]
# X1 = np.random.multivariate_normal(mean1, cov, num_points_per_class)
 
# # Class 2
# mean2 = [10, 0]
# X2 = np.random.multivariate_normal(mean2, cov, num_points_per_class)
 
# # Class 3
# mean3 = [5, 6]
# X3 = np.random.multivariate_normal(mean3, cov, num_points_per_class)




# X = np.concatenate([X1, X2, X3], axis=0)
# X.shape

# def scale_to_01_range(x):
#     # compute the distribution range
#     value_range = (np.max(x) - np.min(x))
 
#     # move the distribution so that it starts from zero
#     # by extracting the minimal value from all its values
#     starts_from_zero = x - np.min(x)
 
#     # make the distribution fit [0; 1] by dividing by its range
#     return starts_from_zero / value_range


# X[:, 0] = scale_to_01_range(X[:, 0])
# X[:, 1] = scale_to_01_range(X[:, 1])


# class VisualizeScatter:
#     def __init__(self, fig_size=(10, 8), xlabel='X', ylabel='Y', title=None, 
#                  size=10, num_classes=3):
#         plt.figure(figsize=fig_size)
#         plt.grid('true')
#         plt.title(title)
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         self.colors = ['red', 'green', 'blue']
#         self.num_classes = num_classes
#         self.size = size
 
#     def add_scatters(self, X):
#         x = X[:, 0]
#         if X.shape[1] == 2:
#             y = X[:, 1]
#         else:
#             y = np.zeros(len(x))
#         points_per_class = len(x) // self.num_classes
#         st = 0
#         end = points_per_class
#         for i in range(self.num_classes):
#             plt.scatter(x[st:end], y[st:end], 
#                 c=self.colors[i % len(self.colors)], 
#                 s=self.size)
#             st = end
#             end = end + points_per_class
 
#     @staticmethod
#     def show_plot():
#         plt.show()
        
        
# vis = VisualizeScatter(fig_size=(10, 8), title="Original Rescaled 2-D Points")
# vis.add_scatters(X)
# vis.show_plot()


# perplexity = 25
# X_embedded = TSNE(n_components=1, 
#     perplexity=perplexity, 
#     learning_rate='auto', 
#     init='random', 
#     random_state=seed).fit_transform(X)
# tsne_vis = VisualizeScatter(fig_size=(10, 2), 
#     title='t-SNE 1-D Projection (perplexity = {})'.format(perplexity))
# tsne_vis.add_scatters(X_embedded)
# tsne_vis.show_plot()



# pca = PCA(n_components=1)
# X_reduced = pca.fit_transform(X)
# pca_vis = VisualizeScatter(fig_size=(10, 2), title='PCA 1-D Projection')
# pca_vis.add_scatters(X_reduced)
# pca_vis.show_plot()

# perplexity = 2
# X_embedded = TSNE(n_components=1, 
#     perplexity=perplexity, 
#     learning_rate='auto', 
#     init='random', 
#     random_state=seed).fit_transform(X)
# tsne_vis = VisualizeScatter(fig_size=(10, 2), 
#     title='t-SNE 1-D Projection (perplexity = {})'.format(perplexity))
# tsne_vis.add_scatters(X_embedded)
# tsne_vis.add_scatters(X_embedded)
# tsne_vis.show_plot()



# perplexity = 150
# X_embedded = TSNE(n_components=1, 
#     perplexity=perplexity, 
#     learning_rate='auto', 
#     init='random', 
#     random_state=seed).fit_transform(X)
 
# tsne_vis = VisualizeScatter(fig_size=(10, 2), 
#     title='t-SNE 1-D Projection (perplexity = {})'.format(perplexity))
# tsne_vis.add_scatters(X_embedded)
# tsne_vis.add_scatters(X_embedded)
# tsne_vis.show_plot()



#%%  

# def download_file(url, save_name):
#     """
#     "Download and save the file."
 
#     arguments:
#     url (str): URL path of the file.
#     save_name: (str): file path to save the downloaded file.
#     """
#     file = requests.get(url)
#     open(save_name, 'wb').write(file.content)
#     print(f"Downloaded {save_name}...")
#     return

# def unzip(zip_file_path=None):
#     """
#     "Unzip the file"
 
#     arguments:
#     zip_file_path (str): The zipped file path
 
#     """
#     try:
#         with ZipFile(zip_file_path) as z:
#             z.extractall("./")
#             print(f"Extracted {zip_file_path}...\n")
#     except:
#         print("Invalid file")
 
#     return


#%%

# if not os.path.exists('animal10'):
#     download_file(
#         'https://www.dropbox.com/sh/wyt8cvctpcvg10r/AAAuOf992j1vDf7S7oV1STW7a?dl=1', 
#         'animal10.zip')
     
#     unzip('animal10.zip')
    
    
#%%    
# def get_classwise_image_path(root_dir):
#     image_paths = dict()
#     classes = os.listdir(root_dir)
#     for cls in classes:
#         image_paths[cls] = []
#         class_dir = os.path.join(root_dir, cls)
#         images = os.listdir(class_dir)
#         for image_name in images:
#             img_path = os.path.join(class_dir, image_name)
#             image_paths[cls].append(img_path)
#     return image_paths


#%%
#IMG_ROOT_DIR = "subset_extract_folder"#'animal10'
#image_paths_dict = get_classwise_image_path(IMG_ROOT_DIR)

annot_file = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/valid/_annotations.coco.json"

img_root = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/subset_extract_folder/valid_subset"
#%%
annot_df = coco_annotation_to_df(coco_annotation_file=annot_file)

#%%
img_props = get_img_names_labels_paths(img_dir=img_root, annot_records_df=annot_df, 
                                        img_ext = ".jpg"
                                        )

#%%

cropped_img_props = crop_image_with_bbox(coco_annotation_file_path=annot_file, 
                                         all_images=False,
                                        images_root_path=img_root, output_dir="cropped_imgs",
                                        image_names=img_props.img_names,
                                        result_store_type = img_props,
                                        save_img_ext=".png",
                                        merge_crop_of_imgs=True
                                        )    
    
#%%

# img_test = img_props.img_names[0]


# #%%
# img_df = annot_df[annot_df["file_name"]==img_test]

# #%%
# ann_id = img_df['id_annotation'].to_list()

# img_item_df = img_df[img_df['id_annotation']==ann_id]
# img_item_bbox = img_item_df['bbox'].to_list()[0]#.values
# x, y, w, h = img_item_bbox

#%%
# import os
# print(os.path.splitext("/path/to/some/file.txt")[0])

# #%%
# os.path.splitext("file.jpg")[0]

#%%
IMG_WIDTH, IMG_HEIGHT = (224, 224)

  
#%%  
def load_and_resize_image(img_path, width, height):
    img = Image.open(img_path).resize((width, height))
    return img


#%%
# def show_class_sample(image_path_dic, fig_size=(15, 6)):
#     fig, axes = plt.subplots(
#         nrows=2,
#         ncols=5,
#         figsize=fig_size
#         )
#     list_axes = list(axes.flat)
#     classes = list(image_path_dic.keys())
#     for i, ax in enumerate(list_axes): 
#         img = load_and_resize_image(image_path_dic[classes[i]][0], 
#             IMG_WIDTH, 
#             IMG_HEIGHT)
#         ax.imshow(img)
#         ax.xaxis.set_visible(False)
#         ax.yaxis.set_visible(False)
#         ax.set_title(classes[i])
#     fig.suptitle("Animal-10 Dataset Samples", fontsize=15)
#     plt.show()
#     return


# show_class_sample(image_paths_dict)


#%%
def load_model_and_preprocess_func(input_shape, model_family, model_name):  
     
    # Models will be loaded wth pre-trainied `imagenet` weights.
    model = getattr(tf.keras.applications, model_name)(input_shape=input_shape, 
        weights="imagenet", 
        include_top=False)
     
    preprocess  = getattr(tf.keras.applications, model_family).preprocess_input
    return model, preprocess



#%%
def get_feature_extractor(model):
    inputs = model.inputs
    x = model(inputs)
    outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
    feat_ext = tf.keras.Model(inputs=inputs, outputs=outputs, 
                                name="feature_extractor"
                            )
    return feat_ext


#%%
IMAGE_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
MODEL_FAMILY = "resnet"
MODEL_NAME   = "ResNet101"
model, preprocess= load_model_and_preprocess_func(IMAGE_SHAPE, 
    MODEL_FAMILY, 
    MODEL_NAME)



feat_ext_model = get_feature_extractor(model)
print(feat_ext_model.summary())



#%%
def extract_features(input, model, preprocess):
     
    # Pre-process the input image.
    x = preprocess(input)
 
    # Generate predictions.
    preds = model.predict(x)
 
    return preds[0]


#%%
def load_image_for_inference(image_path, img_shape):
     
    # Load the image.
    image = tf.io.read_file(image_path)
     
    # Convert the image from bytes to an image tensor.
    x = tf.image.decode_image(image, channels=img_shape[2])
     
    # Resize image to the input shape required by the model.
    x = tf.image.resize(x, (img_shape[0], img_shape[1]))
     
    # Add a dimension for an image batch representation.
    x = tf.expand_dims(x, axis=0)
 
    return x



# def get_images_labels_features(image_paths_dict, feature_extractor, preprocess):
#     images = []
#     labels = []
#     features = []
 
#     for cls in image_paths_dict:
#         image_paths = image_paths_dict[cls]
#         for img_path in image_paths:
#             labels.append(cls)
#             img = load_and_resize_image(img_path, IMG_WIDTH, IMG_HEIGHT)
#             images.append(img)
#             img_for_infer = load_image_for_inference(img_path, IMAGE_SHAPE)
#             feature = extract_features(img_for_infer, 
#                 feature_extractor, 
#                 preprocess)
#             features.append(feature)
#     return images, labels, features


#%%
def get_images_features(img_property_set: ImgPropertySetReturnType,
                        feature_extractor, preprocess, img_width, img_height,
                        use_cropped_imgs: bool = False,
                        use_merged_cropped_imgs: bool = False
                        )->ImgPropertySetReturnType:
    images = []
    features = []
    
    if use_cropped_imgs and use_merged_cropped_imgs:
        raise Error("""Both `use_cropped_imgs` and `use_merged_cropped_imgs` were set to True.
                        Only one of them can be set to True at a time. You can only extract 
                        features from either of them at a time. Hint: Set Either of them to True 
                        and the other to False if you do not want to use full image.
                        By default when both are false, The full image is used for feature extraction.
                    """)
        
    if use_merged_cropped_imgs:
        img_paths = sorted(img_property_set.merged_cropped_img_paths)
    if use_cropped_imgs:
        img_paths = sorted(img_property_set.cropped_img_paths)
    else:
        img_paths = sorted(img_property_set.img_paths)
        
    for img_path in img_paths:
        img = load_and_resize_image(img_path, img_width, img_height)
        images.append(img)
        img_for_infer = load_image_for_inference(img_path, IMAGE_SHAPE)
        feature = extract_features(img_for_infer, feature_extractor, preprocess)
        features.append(feature)
    
    if use_cropped_imgs:
        img_property_set.cropped_imgs = images
        img_property_set.cropped_imgs_features = features
            
    img_property_set.imgs = images
    img_property_set.features = features
    
    return img_property_set
        

#%%
img_feat = get_images_features(img_property_set=img_props,
                               feature_extractor=feat_ext_model,
                               preprocess=preprocess, img_width=IMG_WIDTH,
                               img_height=IMG_HEIGHT, use_cropped_imgs=True
                               )

#%%
if hasattr(img_feat, "cropped_imgs"):
    images = img_feat.cropped_imgs
    features = img_feat.cropped_imgs_features
    labels = img_feat.cropped_img_labels
else:    
    images = img_feat.images
    labels = img_feat.img_labels
    features = img_feat.features








#%%
#images, labels, features = get_images_labels_features(image_paths_dict, feat_ext_model, preprocess)



#%%

clust_eval = clusteval()

#%%

features_np = np.array(features)

#%%
clus_obj = clust_eval.fit(features_np)


#%%

clus_obj.keys()

#%%

#clus_obj['score']

#%%
clusters = clus_obj['labx']


#%%
img_feat.cropped_img_clusters = clusters

#%%
def create_sprite_image(pil_images, save_path):
    # Assuming all images have the same width and height
    img_width, img_height = pil_images[0].size
 
    # create a master square images
    row_coln_count = int(np.ceil(np.sqrt(len(pil_images))))
    master_img_width = img_width * row_coln_count
    master_img_height = img_height * row_coln_count
 
    master_image = Image.new(
        mode = 'RGBA',
        size = (master_img_width, master_img_height),
        color = (0, 0, 0, 0)
    )
 
    for i, img in enumerate(pil_images):
        div, mod = divmod(i, row_coln_count)
        w_loc = img_width * mod
        h_loc = img_height * div
        master_image.paste(img, (w_loc, h_loc))
 
    master_image.convert('RGB').save(save_path, transparency=0)
    return



#%%
# def write_embedding(log_dir, pil_images, features, labels, clusters):
#     """Writes embedding data and projector configuration to the logdir."""
#     metadata_filename = "metadata.tsv"
#     tensor_filename = "features.tsv"
#     sprite_image_filename = "sprite.jpg"
 
 
#     os.makedirs(log_dir, exist_ok=True)
#     with open(os.path.join(log_dir, metadata_filename), "w") as f:
#         for label, cluster in zip(labels, clusters):
#             f.write(f"{label} cluster_{cluster}\n")
#     with open(os.path.join(log_dir, tensor_filename), "w") as f:
#         for tensor in features:
#             f.write("{}\n".format("\t".join(str(x) for x in tensor)))
 
#     sprite_image_path = os.path.join(log_dir, sprite_image_filename)
 
#     config = projector.ProjectorConfig()
#     embedding = config.embeddings.add()
#     # Label info.
#     embedding.metadata_path = metadata_filename
#     # Features info.
#     embedding.tensor_path = tensor_filename
#     # Image info.
#     create_sprite_image(pil_images, sprite_image_path)
#     embedding.sprite.image_path = sprite_image_filename
#     # Specify the width and height of a single thumbnail.
#     img_width, img_height = pil_images[0].size
#     embedding.sprite.single_image_dim.extend([img_width, img_height])
#     # Create the configuration file.
#     projector.visualize_embeddings(log_dir, config)
     
#     return



#%%

def write_embedding(img_property_set: ImgPropertySetReturnType, log_dir: str,
                    use_cropped_imgs: bool = False
                    ):
    """Writes embedding data and projector configuration to the logdir."""
    metadata_filename = "metadata.tsv"
    tensor_filename = "features.tsv"
    sprite_image_filename = "sprite.jpg"
    
    if use_cropped_imgs:
        if not hasattr(img_property_set, "cropped_imgs"):
            raise Error("""img_property_set provided is not that of cropped images while 
                        use_cropped_imgs is set to True. Either set use_cropped_imgs to False
                        to False when you have not cropped images OR crop images and pass the returns
                        of type ImgPropertySetReturnType to img_property_set
                        """
                        )
        else:
            pil_images = img_property_set.cropped_imgs
            features = img_property_set.cropped_imgs_features
            labels = img_property_set.cropped_img_labels
            clusters = img_property_set.cropped_img_clusters
            img_names = img_property_set.cropped_img_names
    else:
        pil_images = img_property_set.imgs
        features = img_property_set.features
        labels = img_property_set.img_labels
        clusters = img_property_set.clusters
        img_names = img_property_set.img_names
    
    os.makedirs(log_dir, exist_ok=True)
    
    with open(os.path.join(log_dir, metadata_filename), "w") as f:
        for label, cluster, img in zip(labels, clusters, img_names):
            #[f.write(f"{i}    \n") for i in label]
            f.write(f"{str(label)[0:-1]} cluster_{cluster}\n")
            #f.write(f"{label} cluster_{cluster}\n")
            #[eg_file.write(f"{i}    ") for i in img_labels]
    with open(os.path.join(log_dir, tensor_filename), "w") as f:
        for tensor in features:
            f.write("{}\n".format("\t".join(str(x) for x in tensor)))
 
    sprite_image_path = os.path.join(log_dir, sprite_image_filename)
 
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # Label info.
    embedding.metadata_path = metadata_filename
    # Features info.
    embedding.tensor_path = tensor_filename
    # Image info.
    create_sprite_image(pil_images, sprite_image_path)
    embedding.sprite.image_path = sprite_image_filename
    # Specify the width and height of a single thumbnail.
    img_width, img_height = pil_images[0].size
    embedding.sprite.single_image_dim.extend([img_width, img_height])
    # Create the configuration file.
    projector.visualize_embeddings(log_dir, config)
     
    return

#%%
LOG_DIR = os.path.join('cropped_logs', MODEL_NAME)

#%%

write_embedding(img_property_set=img_feat, log_dir=LOG_DIR,
                use_cropped_imgs=True
                )


#%%

#img_feat.cropped_img_labels

#%%
#li = ['Mosaic Virus', 'Mosaic Virus', 'Mosaic Virus', 'Mosaic Virus']

#%%

#str(li)[1:-1]
#%%
#f"{item }"


#%%
#write_embedding(LOG_DIR, images, features, labels, clusters)


#%%
#load_ext tensorboard

#%%
# %reload_ext tensorboard
notebook.start(f"--logdir {LOG_DIR} --port default") 
#tensorboard --logdir {LOG_DIR} --port 8090





        

# %%
#%%
import matplotlib.pyplot as plt
import cv2


#%%
plt.imshow(np.array(images[50]))#.shape
#%%
features[0].shape
#%%

feat_list = [features[0][:], features[1][:], features[2][:], features[3][:]]

#%%

len(feat_list)
#%%
feat_array = np.array(feat_list)

#%%
feat_array.shape

#%%

feat_concat = np.concatenate((features[0], features[1], features[2], features[3])).reshape(-1,1)#.shape


#%%
feat_concat.shape
#%%

len(feat_list)
#%%

pca = PCA(n_components=3)

pca.fit_transform(feat_concat)


#%%

#%%
import pywt
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#%% This function does the coefficient fusing according to the fusion method
def fuseCoeff(cooef1, cooef2, method):

    if (method == 'mean'):
        cooef = (cooef1 + cooef2) / 2
    elif (method == 'min'):
        cooef = np.minimum(cooef1,cooef2)
    elif (method == 'max'):
        cooef = np.maximum(cooef1,cooef2)
    else:
        cooef = []

    return cooef


# Params
FUSION_METHOD = 'mean' # Can be 'min' || 'max || anything you choose according theory

#%% Read the two image
img_path1 = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/cropped_imgs/67.0_cropped_62EggPlant_Mosaic_jpg.rf.19e9a3fb54af6e0aac7ae9ad00c125dd.png"
img_path2 = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/cropped_imgs/68.0_cropped_62EggPlant_Mosaic_jpg.rf.19e9a3fb54af6e0aac7ae9ad00c125dd.png"
I1 = cv2.imread(img_path1)
I2 = cv2.imread(img_path2)

I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)
#plt.imshow(img1)
#I1 = plt.imread(img_path1)
#I2 = plt.imread(img_path2)

#%%

I1.shape

#%%
I2.shape

#%% We need to have both images the same size
I2 = cv2.resize(I2,(I1.shape[1], I1.shape[0])) # I do this just because i used two random images

## Fusion algo

#%%

plt.imshow(I2)

#%%
plt.imshow(I1)

#%% First: Do wavelet transform on each image
wavelet = 'db1'
cooef1 = pywt.wavedec2(images[0], wavelet)
cooef2 = pywt.wavedec2(images[1], wavelet)

# Second: for each level in both image do the fusion according to the desire option
fusedCooef = []
for i in range(len(cooef1)-1):

    # The first values in each decomposition is the apprximation values of the top level
    if(i == 0):

        fusedCooef.append(fuseCoeff(cooef1[0],cooef2[0],FUSION_METHOD))

    else:

        # For the rest of the levels we have tupels with 3 coeeficents
        c1 = fuseCoeff(cooef1[i][0],cooef2[i][0],FUSION_METHOD)
        c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], FUSION_METHOD)
        c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], FUSION_METHOD)

        fusedCooef.append((c1,c2,c3))

# Third: After we fused the cooefficent we nned to transfor back to get the image
fusedImage = pywt.waverec2(fusedCooef, wavelet)

# Forth: normmalize values to be in uint8
fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
fusedImage = fusedImage.astype(np.uint8)

#%%
fusedImage.shape

#%%


# define a function for horizontally 
# concatenating images of different 
# heights 
def merge_imgs(img_list, 
				interpolation = cv2.INTER_CUBIC,
                merge_type="horizontal"):
    img_array_list = [np.array(img) for img in img_list]

    if merge_type == "horizontal":
        h_min = min(img.shape[0] for img in img_array_list) 
        im_list_resize = [cv2.resize(img, 
					(int(img.shape[1] * h_min / img.shape[0]), 
						h_min), interpolation 
								= interpolation) 
					for img in img_array_list]
        return cv2.hconcat(im_list_resize)
    else:
        h_min = min(img.shape[1] for img in img_array_list)
        im_list_resize = [cv2.resize(img, 
                        (int(img.shape[0] * h_min / img.shape[1]), 
                            h_min), interpolation 
                                    = interpolation) 
                        for img in img_array_list] 
    return cv2.vconcat(im_list_resize)



#%%
# define a function for vertically 
# concatenating images of different 
# widths 
def vconcat_resize(img_list, interpolation 
				= cv2.INTER_CUBIC): 
	# take minimum width 
	w_min = min(img.shape[1] 
				for img in img_list) 
	
	# resizing images 
	im_list_resize = [cv2.resize(img, 
					(w_min, int(img.shape[0] * w_min / img.shape[1])), 
								interpolation = interpolation) 
					for img in img_list] 
	# return final image 
	return cv2.vconcat(im_list_resize) 

# function calling 
# img_v_resize = vconcat_resize([img1, img2, img1]) 

# # show the output image 
# cv2.imwrite('vconcat_resize.jpg', img_v_resize) 

#%%

#%% function calling 
img_h_resize = merge_imgs([images[10], images[2], images[13]], merge_type="vertical") 

plt.imshow(img_h_resize)

#%%
img_h_resize.shape


#%% show the Output image 
cv2.imshow('hconcat_resize.jpg', img_h_resize) 


