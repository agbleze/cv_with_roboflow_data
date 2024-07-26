
#%%
from cluster_utils import coco_annotation_to_df
import os
import argparse 
import inspect


#%%

                    
def convert_coco_to_kitti(output_dir, coco_file_path, *args, **kwargs):#(output_dir, coco_file_path: str):
    coco_df_params = inspect.signature(coco_annotation_to_df).parameters.keys()
    filtered_args = {k: v for k, v in zip(coco_df_params, args) if k in coco_df_params}
    print(filtered_args)
    os.makedirs(output_dir, exist_ok=True)
    annot_df = coco_annotation_to_df(**filtered_args)#(coco_file_path)                    
    for img in annot_df['file_name'].unique():
        img_df = annot_df[annot_df['file_name']==img]
        ann_ids = img_df['id_annotation'].values
        img_name_without_ext = img.split('.')[0]
        output_file = os.path.join(output_dir, img_name_without_ext)
        with open(output_file + '.txt','w') as label_file:
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
    parser = argparse.ArgumentParser(description="A helper for converting coco to kitti")
    parser.add_argument("--output_dir", required=True,
                      type=str,
                      help="Output directory to write kitti labels to")
    parser.add_argument("--coco_file_path", required=True,
                      type=str,
                      help="Path to coco annotation file."
                      )
    
    args = parser.parse_args()
    convert_coco_to_kitti(output_dir=args.output_dir, 
                          coco_file_path=args.coco_file_path)
        
# %%
from inspect import signature
def foo(a, *, b:int, **kwargs):
    pass

sig = signature(foo)

str(sig)

#%%
str(sig.parameters['b'])


sig.parameters['b'].annotation

#%%
for i in sig.parameters.keys():
    print(i)
# %%
