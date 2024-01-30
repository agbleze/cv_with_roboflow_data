

#%%
import json



#%%

specfile = "/home/lin/cv_with_roboflow_data/nvidia_tao_tut/default_specfile.json"

with open(specfile, "r") as spec:
    data = json.loads(spec.read())
# %%
data.items()

#%%
nv_file = data.get("nvidia_specfile")
for key in nv_file.keys():
    if type(nv_file[key]) == dict:
        print(key)
    else:
        print(f"{key}: {nv_file[key]}")


#%%
infer_list = []
for it in nv_file["inference_config"].items():
    infer_set = f"{it[0]}: {it[1]}\n"
    infer_list.append(infer_set)
    #print(it)

#%%    
print(*infer_list)

#%%
inference_config = f"""inference_config: {{
      {"".join(infer_list)}
      }}
      """
      
print(inference_config)



#%%
def get_unnested_setting(json_file, nvidia_specfile_key: str ="nvidia_specfile",
                         config_type: str="inference_config"):
    nv_file_json = json_file.get(nvidia_specfile_key)
    infer_list = []
    for it in nv_file_json[config_type].items():
        infer_set = f"{it[0]}: {it[1]}\n"
        infer_list.append(infer_set)
        
    config = f"""{config_type}: {{
                            {"".join(infer_list)}
                            }}
                            """
    return config
        
        
#%%

infer_config = get_unnested_setting(json_file=data)

print(infer_config)  

#%%
evaluate_config = get_unnested_setting(json_file=data, config_type="evaluation_config")

print(evaluate_config)


#%%

if type(nv_file["training_config"]) == dict:
    train_dict = nv_file["training_config"]
    first_cl = [f"{key}: {train_dict[key]}\n" for key in train_dict.keys() 
                if type(train_dict[key]) != dict
                ]
    # nest_cl = [f"""classifier_regr_std {{
    #            "key": {item[0]}
    #            "value": {item[1]}
    #            }}
    #            """
    #            for item in train_dict["classifier_regr_std"].items()

               
    #             #for key in train_dict.keys() 
    #             #if (type(train_dict[key]) == dict) and (train_dict.keys() == "classifier_regr_std")
    #             ]
    
    # ls_list = []
    for key in train_dict.keys():
        if type(train_dict[key]) == dict:
            n_dict = train_dict[key]
            for key_s in n_dict.keys():
                if type(n_dict[key_s]) != dict:
                    if key_s in ["x", "y", "w", "h"]:
                        n_dict_items = train_dict[key].items()
                        sep_list = [f"""\n{key} {{\n
                                    "key": {n_dict_item[0]}
                                    "value": {n_dict_item[1]}
                                    }}
                                    """#.strip()
                                    for n_dict_item in n_dict_items
                                    ]
                        sep_list_fr = "".join(sep_list)
                    else:
                        n_dict_items = train_dict[key].items()
                        all_items_n = [f"{n_dict_item[0]}: {n_dict_item[1]}\n" for n_dict_item in n_dict_items]
                        all_items_n_fr = "".join(all_items_n)
                        all_items_fn = f"\n{key} {{\n {all_items_n_fr} }}"
        #ls_list.append(all_items_fn)
                        
                        
        
    each_item_nested = []
    #d_item_nested = []
    more_nested_list = []
    for keys in train_dict.keys():
        #if (type(train_dict[keys]))!=dict:
        #    pass
        if (type(train_dict[keys]))==dict: 
            sec_train_dict = train_dict[keys]
            for key_c in sec_train_dict.keys():
                if type(sec_train_dict[key_c]) == dict:
                    sec_items = sec_train_dict[key_c].items()
                    #print(sec_items)
                    #for item in sec_items:
                    key_c_list = []
                    for item in sec_items:
                        
                        if type(item[1]) != list:
                            nested_item= f"{item[0]}: {item[1]}\n"#.strip()
                            key_c_list.append(nested_item)
                            #each_item_nested.append(nested_item)
                        elif type(item[1]) == list:
                            #nnested_list_item = []
                            for i in range(len(item[1])):
                                nnested_item = f"{item[0]}: {item[1][i]}\n"
                                #each_item_nested.append(nnested_item)
                                #nnested_list_item.append(nnested_item)
                                key_c_list.append(nnested_item)
                    #each_item_nested.extend(key_c_list)
                                
            format_each_item_nested = "".join(key_c_list)#.strip()
            if keys not in ['classifier_regr_std', 'regularizer']:
                more_nested = f"""{keys}{{\n
                    {key_c}{{\n
                    {format_each_item_nested}
                    }}
                }}\n"""#.strip() 
                more_nested_list.append(more_nested)      

                more_nested_list_set = "".join(more_nested_list)
                    
    first_cl_set = "".join(first_cl)       
    nest_set = "".join(nest_cl)#.strip()
    
    training_set = f"""training_config {{\n
                    {first_cl_set}
                    {sep_list_fr}
                    {all_items_fn}
                    {more_nested_list_set}
                    
                    }}
                    """
    print(training_set)
    print("\n \n")
    #print(ls_list)
    #print(first_cl_set)
    #print("nested_set \n")
    #print(nest_set)
    #print("\n more nested")
    #print(more_nested_list_set)
    #print(nest_set)



#%% ##########   model_config   ###############
from collections import Iterable
model_dict = nv_file["model_config"]
if type(model_dict) == dict:
    for key in model_dict.keys():
        if type(model_dict[key]) != dict:
            model_items = model_dict.items()
            base_list = []
            for item in model_items:
                if (not isinstance(item[1], list)) and (not isinstance(item[1], dict)):
                    if isinstance(item[1], str):
                      base_pair = f"\n{item[0]}: '{item[1]}' \n"  
                    base_pair = f"\n{item[0]}: {item[1]} \n"
                    base_list.append(base_pair)
                elif isinstance(item[1], list):
                    for i in range(len(item[1])):
                                base_pair_nested = f"{item[0]}: {item[1][i]}\n"
                                #each_item_nested.append(nnested_item)
                                #nnested_list_item.append(nnested_item)
                                base_list.append(base_pair_nested)
    base_fr = "".join(base_list)
    
    ##### work on second order dict   
    for key in model_dict.keys():
        if isinstance(model_dict[key], dict):
            #dict_sec = model_dict[key]
            #for key_sec in dict_sec.keys():
            if key == "input_image_config":
                img_config_dict = model_dict["input_image_config"]
                img_config_items = img_config_dict.items()
                #for item in img_config_items:
                    #if not isinstance(item[1], dict):
                it_list = [f"\n{item[0]}: {item[1]}\n" 
                            for item in img_config_items 
                            if not isinstance(item[1], dict)
                            ]
                it_fr = "".join(it_list)
                if isinstance(item[1], dict):
                    if item[0] != "image_channel_mean":
                        #item_dict_item_list = []
                        item_dict_items = item[1].items()
                        item_dict_item_list = [f"{itm[0]}: {itm[1]}\n" for itm in item_dict_items]
                        item_dict_item_fr = "".join(item_dict_item_list)
                        it_frstr = f"""{item[0]}{{\n
                        {item_dict_item_fr}
                        }}
                        """
                    elif item[0] == "image_channel_mean":
                        img_chn_it = item[1].items()
                        img_chn_item_list = [f"""{item[0]}{{\n
                            key: '{img_it[0]}'\n
                            value: {img_it[1]}
                            }}\n""" for img_it in img_chn_it
                            ]
                        img_chn_item_fr = "".join(img_chn_item_list)
            
                input_img_config = f"""{key} {{\n
                {it_fr}
                {it_frstr}
                {img_chn_item_fr}
                    }}"""
                    
        for key in model_dict.keys():
            if isinstance(model_dict[key], dict):
                items = model_dict[key].items()
                nested_list_in_dict = []
                #if isinstance(item[1], list):
                    
                for item in items:
                    if not isinstance(item[1], list):
                        key_val_pair_in_dict = [f"\n{item[0]}: {item[1]}\n" for item in items]
                        key_val_pair_in_dict_fr = "".join(key_val_pair_in_dict) 
                    key_val_pair_in_dict_output = f"""\n{key} {{\n
                    {key_val_pair_in_dict_fr}
                    }}"""
                        
        #for key in model_dict.keys():
            # if isinstance(model_dict[key], dict):
            #     items = model_dict[key].items()
            #     nested_list_in_dict = [] 
            #     for item in items:               
                    if isinstance(item[1], list):
                        for i in range(len(item[1])):
                            key_list_pair = f"\n{item[0]}: {item[1][i]}\n" 
                            #key_list_pair_ap = 
                            nested_list_in_dict.append(key_list_pair)
                            #print(key_list_pair_ap)
                        key_list_pair_fr = "".join(nested_list_in_dict)
                        key_list_pair_output = f"""\n{key} {{\n
                        {key_list_pair_fr}
                        }}"""
                        #key_list_pair_fr_ = f"""{}"""
                        
                            #each_item_nested.append(nnested_item)
                            #nnested_list_item.append(nnested_item)
                            #nested_list_in_dict.append(key_list_pair)
            # else:
            #     key_val_pair_in_dict = [f"\n{item[0]}: {item[1]}\n" for item in items]
            #     key_val_pair_in_dict_fr = "".join(key_val_pair_in_dict)  
                #nested_list_in_dict.append(key_val_pair_in_dict_fr) 
                    
                #nested_list_in_dict_fr = "".join(nested_list_in_dict)   
        key_val_list_in_dict_fr = f"""{key_val_pair_in_dict_output}
        {key_list_pair_output}
        """          
                    
            
        
    # model_first_cl = [f"\n{key}: {model_dict[key]}\n" for key in model_dict.keys() 
    #             if type(model_dict[key]) != dict
    #             ]
    # model_first_cl_set = "".join(model_first_cl)
    
    model_set = f"""model_config {{
    {base_fr}
    {input_img_config}
    {key_val_list_in_dict_fr}
                    
    }}
                    """
    print(model_set)



##############  TO DO: parse  dataset_config  #############
#%%  dataset_config

data_dict = nv_file["dataset_config"]

if type(data_dict) == dict:
    for key in data_dict.keys():
        if not isinstance(data_dict[key], dict):
            data_items = data_dict.items()
            data_base_list = []
            for item in data_items:
                if (not isinstance(item[1], list)) and (not isinstance(item[1], dict)):
                    if isinstance(item[1], str):
                      data_base_pair = f"\n{item[0]}: '{item[1]}' \n"  
                    data_base_pair = f"\n{item[0]}: {item[1]} \n"
                    data_base_list.append(data_base_list)
                elif isinstance(item[1], list):
                    for i in range(len(item[1])):
                                data_base_pair_nested = f"{item[0]}: {item[1][i]}\n"
                                data_base_list.append(data_base_pair_nested)
    data_base_fr = "".join(data_base_list)
    print(data_base_fr)



#%%
for item in data_dict.items():
    #print(data_dict[key])
    first_obj_list = []
    if not isinstance(item[1], dict):
        first_fr = f"\n{item[0]}: {item[1]}\n"
        print(data_dict.items())
        #f"{key}: "
    

#%%
fir_obj = [f"\n{item[0]}: {item[1]}\n" for item in data_dict.items() if not isinstance(item[1], dict)]
first_output = "".join(fir_obj)

for item in data_dict.items():
    if isinstance(item[1], dict):
        if item[0] != "target_class_mapping":
            #item_dict_item_list = []
            data_dict_items = item[1].items()
            data_dict_item_list = [f"\n{itm[0]}: {itm[1]}\n" for itm in data_dict_items]
            data_dict_item_fr = "".join(data_dict_item_list)
            it_frstr = f"""{item[0]}: {{\n
            {data_dict_item_fr}
            }}
            """
        elif item[0] == "target_class_mapping":
            target_class_it = item[1].items()
            target_class_item_list = [f"""\n{item[0]}{{\n
            key: '{target_it[0]}'\n
            value: '{target_it[1]}'
            }}\n""" for target_it in target_class_it
                ]
            target_class_item_fr = "".join(target_class_item_list)

# input_img_config = f"""{key} {{\n
# {it_fr}
# {it_frstr}
# {target_class_item_fr}
#     }}"""
        
dataset_config = f"""dataset_config {{\n
{first_output}
{it_frstr}
{target_class_item_fr}
}}"""
print(dataset_config)



#%%
######## augmentation_config ##########
for key in nv_file.keys():
    if key == "augmentation_config":
        augment_dict = nv_file["augmentation_config"]
        augment_dict_items = augment_dict.items()
        aug_itm_list = []
        for item in augment_dict_items:
            if isinstance(item[1], dict):
                aug_it_list = [f"{aug_it[0]}: {aug_it[1]}\n" 
                               for aug_it in item[1].items()
                               ]
                aug_it_fr = "".join(aug_it_list)
                aug_itm = f"{item[0]} {{\n {aug_it_fr} \n}}\n"
                aug_itm_list.append(aug_itm)
        aug_conf_fr = "".join(aug_itm_list)
        aug_config_output = f"augmentation_config {{\n {aug_conf_fr} }}"
        print(aug_config_output)
                
        


#%%
nv_file = data.get("nvidia_specfile")
for key in nv_file.keys():
    if type(nv_file[key]) == dict:
        print(key)
    else:
        print(f"{key}: {nv_file[key]}")  
#%%
"jkaj"

# %%
