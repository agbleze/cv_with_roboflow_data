
#%%
from cluster_utils import coco_annotation_to_df
import os
from argparse import Argument

                    
def convert_coco_to_kitti(output_dir, coco_file_path: str):
    os.makedirs(output_dir, exist_ok=True)
    annot_df = coco_annotation_to_df(coco_file_path)                    
    for img in annot_df['file_name'].unique():
        img_df = annot_df[annot_df['file_name']==img]
        ann_ids = img_df['id_annotation'].values
        img_name_without_ext = img.split('.')[0]
        with open(output_dir + img_name_without_ext + '.txt','w') as label_file:
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
                    
            

if __name__ == '__main__':
    args = Argument(description="A helper for converting coco to kitti")
    args.add_argument("--output", required=True,
                      type=str,
                      help="Output directory to write kitti labels to")
    args.add_argument("--coco_file_path", required=True,
                      type=str,
                      help="Path to coco annotation file."
                      )
    
    convert_coco_to_kitti(output_dir=args.output_dir, coco_file_path=args.coco_file_path)
        
# %%
