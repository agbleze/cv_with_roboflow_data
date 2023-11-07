
#%%
from cluster_utils import coco_annotation_to_df
# %%
annot_file = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/Tomato-pest&diseases-1/valid/_annotations.coco.json"

img_root = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/subset_extract_folder/valid_subset"


#%%
annot_df = coco_annotation_to_df(annot_file)

#%%

annot_df

#%%

img_df = annot_df[annot_df['file_name']==img]

ann_ids = img_df['id_annotation'].values#[2]


#%%

ann_df = img_df[img_df['id_annotation']==ann_id]
#bbox = 
ann_df['bbox'].tolist()[0][2]

#%%

def coco2kitti(catNms, annFile):

    # initialize COCO api for instance annotations
    coco = COCO(annFile)

    # Create an index for the category names
    cats = coco.loadCats(coco.getCatIds())
    cat_idx = {}
    for c in cats:
        cat_idx[c['id']] = c['name']

    for img in coco.imgs:

        # Get all annotation IDs for the image
        catIds = coco.getCatIds(catNms=catNms)
        annIds = coco.getAnnIds(imgIds=[img], catIds=catIds)

        # If there are annotations, create a label file
        if len(annIds) > 0:
            # Get image filename
            img_fname = coco.imgs[img]['file_name']
            # open text file
            with open('./labels/' + img_fname.split('.')[0] + '.txt','w') as label_file:
                anns = coco.loadAnns(annIds)
                for a in anns:
                    bbox = a['bbox']
                    # Convert COCO bbox coords to Kitti ones
                    bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
                    bbox = [str(b) for b in bbox]
                    catname = cat_idx[a['category_id']]
                    # Format line in label file
                    # Note: all whitespace will be removed from class names
                    out_str = [catname.replace(" ","")
                               + ' ' + ' '.join(['0']*2)
                               + ' ' + ' '.join([b for b in bbox])
                               + ' ' + ' '.join(['0']*8)
                               +'\n']
                    label_file.write(out_str[0])
                    
                    
                    
# %%
for img in annot_df['file_name'].unique():
    img_df = annot_df[annot_df['file_name']==img]
    ann_ids = img_df['id_annotation'].values
    img_name_without_ext = img.split('.')[0]
    with open('./test_labels/' + img_name_without_ext + '.txt','w') as label_file:
        for ann_id in ann_ids:
            ann_df = img_df[img_df['id_annotation']==ann_id]
            bbox = ann_df['bbox'].tolist()[0] #ann_df['bbox'][0]#.values 
            bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
            bbox = [str(b) for b in bbox]
            obj_name = ann_df['category_name'].tolist()[0]
            
            out_str = [obj_name.replace(" ","")
                               + ' ' + ' '.join(['0']*2)
                               + ' ' + ' '.join([b for b in bbox])
                               + ' ' + ' '.join(['0']*8)
                               +'\n']
            label_file.write(out_str[0])
                
        
    
# %%
