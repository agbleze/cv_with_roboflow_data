
import dask
from dask.distributed import Client, progress




from cluster_utils import coco_annotation_to_df
import os
import argparse 
import inspect


#%%
coco_file_path = "/home/lin/cv_with_roboflow_data/Tomato-pest&diseases-1/train/_annotations.coco.json"
annot_df = coco_annotation_to_df(coco_file_path) 

imgs = annot_df['file_name'].unique()
output_dir="/home/lin/cv_with_roboflow_data/dask_kitti"
params = [(output_dir, annot_df, img) for img in imgs]
                    
def convert_coco_to_kitti(output_dir, annot_df, img, *args, **kwargs):#(output_dir, coco_file_path: str):
    os.makedirs(output_dir, exist_ok=True)                   
    #for img in annot_df['file_name'].unique():
    img_df = annot_df[annot_df['file_name']==img]
    ann_ids = img_df['id_annotation'].values
    img_name_without_ext = img.split('.')[0]
    output_file = os.path.join(output_dir, img_name_without_ext)
    with open(output_file + '.txt','w') as label_file:
        
        print(img)
        for ann_id in ann_ids:
            ann_df = img_df[img_df['id_annotation']==ann_id]
            bbox = ann_df['bbox'].tolist()#
            print(f"len of bbox is {len(bbox)}")
            bbox = bbox[0] #ann_df['bbox'][0]#.values 
            bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
            bbox = [str(b) for b in bbox]
            obj_name = ann_df['category_name'].tolist()[0]
            
            out_str = [obj_name.replace(" ","")
                            + ' ' + ' '.join(['0']*2)
                            + ' ' + ' '.join([b for b in bbox])
                            + ' ' + ' '.join(['0']*8)
                            +'\n']
            label_file.write(out_str[0])
                


                
def convert_parallel(output_dir, coco_file_path):
    annot_df = coco_annotation_to_df(coco_file_path) 

    imgs = annot_df['file_name'].unique()
    params = [(output_dir, annot_df, img) for img in imgs]
    client = Client(threads_per_worker=4, n_workers=1)
    #client.cluster.scale(2)
    futures = []

    for param in params:
        #dask_res = client.submit(convert_coco_to_kitti, *param)
        dask_res = dask.delayed(convert_coco_to_kitti)(*param)
        futures.append(dask_res)

    #res = client.gather(futures)
    #return res
    dask.compute(*futures)
    
    
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
    res = convert_parallel(output_dir=args.output_dir, coco_file_path=args.coco_file_path)
    #convert_coco_to_kitti(output_dir=args.output_dir, 
    #                      coco_file_path=args.coco_file_path)
        


