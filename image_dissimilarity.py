
#%%
import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

def compute_dissimilarity_scores(folder_path):
    # Load VGG16 model pre-trained on ImageNet
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    
    # Get list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png', 'bmp'))]
    
    # Extract features for each image
    features_list = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        features = extract_features(img_path, model)
        features_list.append(features)
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(features_list)
    
    # Compute dissimilarity scores (1 - cosine similarity)
    dissimilarity_matrix = 1 - similarity_matrix
    
    # Create a DataFrame from the dissimilarity scores
    dissimilarity_scores = []
    for i, img1_file in enumerate(image_files):
        dissimilarity_score = np.mean(dissimilarity_matrix[i])
        dissimilarity_scores.append((img1_file, dissimilarity_score))
    
    df = pd.DataFrame(dissimilarity_scores, columns=['Image', 'DissimilarityScore'])
    
    return df

#%% Example usage
folder_path = '/home/lin/codebase/cv_with_roboflow_data/Tomato-pest&diseases-1/test'
df_dissimilarity = compute_dissimilarity_scores(folder_path)
print(df_dissimilarity)




# %%
