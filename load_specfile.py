

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
        
    config = f"""{config_type}: {{ \n{"".join(infer_list)}\n}} """
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
    for key in train_dict.keys():
        if type(train_dict[key]) == dict:
            n_dict = train_dict[key]
            for key_s in n_dict.keys():
                if type(n_dict[key_s]) != dict:
                    if key_s in ["x", "y", "w", "h"]:
                        n_dict_items = train_dict[key].items()
                        sep_list = [f"""\n{key} {{\n"key": {n_dict_item[0]}\n"value": {n_dict_item[1]}\n}}"""#.strip()
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
                                nnested_item = f"\n{item[0]}: {item[1][i]}\n"
                                #each_item_nested.append(nnested_item)
                                #nnested_list_item.append(nnested_item)
                                key_c_list.append(nnested_item)
                    #each_item_nested.extend(key_c_list)
                                
                    format_each_item_nested = "".join(key_c_list)#.strip()
            if keys not in ['classifier_regr_std', 'regularizer']:
                more_nested = f"""\n{keys}{{\n{key_c}{{\n{format_each_item_nested}\n}}}}\n"""#.strip() 
                more_nested_list.append(more_nested)      

                more_nested_list_set = "".join(more_nested_list)
                    
        first_cl_set = "".join(first_cl)
    
    training_set = f"""training_config {{\n
                    {first_cl_set}
                    {sep_list_fr}
                    {all_items_fn}
                    {more_nested_list_set}
                    
                    }}
                    """
    print(training_set)
    print("\n \n")


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
                                base_pair_nested = f"\n{item[0]}: {item[1][i]}\n"
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
                for item in img_config_items:
                    if isinstance(item[1], dict):
                        if item[0] != "image_channel_mean":
                            #item_dict_item_list = []
                            item_dict_items = item[1].items()
                            item_dict_item_list = [f"\n{itm[0]}: {itm[1]}\n" for itm in item_dict_items]
                            item_dict_item_fr = "".join(item_dict_item_list)
                            it_frstr = f"""{item[0]} {{\n{item_dict_item_fr}\n}}"""
                        elif item[0] == "image_channel_mean":
                            img_chn_it = item[1].items()
                            img_chn_item_list = [f"""{item[0]}{{\n key: '{img_it[0]}'\n value: {img_it[1]} \n}}\n""" for img_it in img_chn_it]
                            img_chn_item_fr = "".join(img_chn_item_list)
                
                            input_img_config = f"""\n{key} {{\n{it_fr}\n{it_frstr}\n{img_chn_item_fr}\n}}"""
                    
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
        key_val_list_in_dict_fr = f"""{key_val_pair_in_dict_output}
        {key_list_pair_output}
        """          
    
    model_set = f"""model_config {{{base_fr}\n{input_img_config}\n{key_val_list_in_dict_fr} \n}}"""
    print(model_set)



##############  TO DO: parse  dataset_config  #############
#%%  dataset_config

data_dict = nv_file["dataset_config"]



#%%
for item in data_dict.items():
    #print(data_dict[key])
    first_obj_list = []
    if not isinstance(item[1], dict):
        first_fr = f"\n{item[0]}: {item[1]}\n"
        print(data_dict.items())
        #f"{key}: "
    

#%%  ######### dataset_config  ###########
fir_obj = [f"\n{item[0]}: {item[1]}\n" for item in data_dict.items() if not isinstance(item[1], dict)]
first_output = "".join(fir_obj)

dict_item_list = []
for item in data_dict.items():
    if isinstance(item[1], dict):
        
        if item[0] != "target_class_mapping":
            #item_dict_item_list = []
            data_dict_items = item[1].items()
            data_dict_item_list = [f"\n{itm[0]}: {itm[1]}\n" for itm in data_dict_items]
            data_dict_item_fr = "".join(data_dict_item_list)
            it_frstr = f"""\n{item[0]}: {{\n{data_dict_item_fr}\n }}"""
            dict_item_list.append(it_frstr)
            #print(it_frstr)
        elif item[0] == "target_class_mapping":
            target_class_it = item[1].items()
            target_class_item_list = [f"""\n{item[0]}{{\n key: '{target_it[0]}'\n value: '{target_it[1]}'}}\n""" 
                                      for target_it in target_class_it
                                    ]
            target_class_item_fr = "".join(target_class_item_list)
        it_frstr = "".join(dict_item_list)

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

def generate_augmentation_config(json_file, nvidia_specfile_key: str ="nvidia_specfile",
                         config_type: str="augmentation_config"
                         ):
    nv_file_json = json_file.get(nvidia_specfile_key)
    nv_file_keys = nv_file_json.keys()
    if config_type not in nv_file_keys:
        raise Exception(f"""The config_type parameter {config_type} is 
                        not a key in the specfile. Ensure that {config_type} 
                        is a key in the json file
                        """
                        ) 
    augment_dict = nv_file_json[config_type]
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
    aug_config_output = f"{config_type} {{\n {aug_conf_fr} }}"
    print(aug_config_output)
    return aug_config_output
        
#%%  ############ break down specfile generation into function   ####################

def generate_dataset_config(json_file, nvidia_specfile_key: str ="nvidia_specfile",
                         config_type: str="dataset_config"
                         ):
    nv_file_json = json_file.get(nvidia_specfile_key)
    nv_file_keys = nv_file_json.keys()
    if config_type not in nv_file_keys:
        raise Exception(f"""The config_type parameter {config_type} is 
                        not a key in the specfile. Ensure that {config_type} 
                        is a key in the json file
                        """
                        )
    data_dict = nv_file_json[config_type]
    fir_obj = [f"\n{item[0]}: {item[1]}\n" for item in data_dict.items() if not isinstance(item[1], dict)]
    first_output = "".join(fir_obj)

    dict_item_list = []
    for item in data_dict.items():
        if isinstance(item[1], dict): 
            if item[0] != "target_class_mapping":
                data_dict_items = item[1].items()
                data_dict_item_list = [f"\n{itm[0]}: {itm[1]}\n" for itm in data_dict_items]
                data_dict_item_fr = "".join(data_dict_item_list)
                it_frstr = f"""\n{item[0]}: {{\n{data_dict_item_fr}\n }}"""
                dict_item_list.append(it_frstr)
            elif item[0] == "target_class_mapping":
                target_class_it = item[1].items()
                target_class_item_list = [f"""\n{item[0]}{{\n key: '{target_it[0]}'\n value: '{target_it[1]}'}}\n""" 
                                        for target_it in target_class_it
                                        ]
                target_class_item_fr = "".join(target_class_item_list)
            it_frstr = "".join(dict_item_list)
            
    dataset_config = f"""{config_type} {{\n{first_output}\n{it_frstr}\n{target_class_item_fr}\n}}"""
    print(dataset_config)
    return dataset_config
    


#%%  #########  model_config   ###############
def generate_model_config(json_file, nvidia_specfile_key: str ="nvidia_specfile",
                         config_type: str="model_config"
                         ):
    nv_file_json = json_file.get(nvidia_specfile_key)
    nv_file_keys = nv_file_json.keys()
    if config_type not in nv_file_keys:
        raise Exception(f"""The config_type parameter {config_type} is 
                        not a key in the specfile. Ensure that {config_type} 
                        is a key in the json file
                        """
                        )
    model_dict = nv_file[config_type]
    if isinstance(model_dict, dict):
        for key in model_dict.keys():
            if not isinstance(model_dict, dict):
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
                            base_pair_nested = f"\n{item[0]}: {item[1][i]}\n"
                            base_list.append(base_pair_nested)
        base_fr = "".join(base_list)
        
        ##### work on second order dict   
        for key in model_dict.keys():
            if isinstance(model_dict[key], dict):
                if key == "input_image_config":
                    img_config_dict = model_dict["input_image_config"]
                    img_config_items = img_config_dict.items()
                    it_list = [f"\n{item[0]}: {item[1]}\n" 
                                for item in img_config_items 
                                if not isinstance(item[1], dict)
                                ]
                    it_fr = "".join(it_list)
                    for item in img_config_items:
                        if isinstance(item[1], dict):
                            if item[0] != "image_channel_mean":
                                item_dict_items = item[1].items()
                                item_dict_item_list = [f"\n{itm[0]}: {itm[1]}\n" for itm in item_dict_items]
                                item_dict_item_fr = "".join(item_dict_item_list)
                                it_frstr = f"""{item[0]} {{\n{item_dict_item_fr}\n}}"""
                            elif item[0] == "image_channel_mean":
                                img_chn_it = item[1].items()
                                img_chn_item_list = [f"""{item[0]}{{\n key: '{img_it[0]}'\n value: {img_it[1]} \n}}\n""" for img_it in img_chn_it]
                                img_chn_item_fr = "".join(img_chn_item_list)
                    
                                input_img_config = f"""\n{key} {{\n{it_fr}\n{it_frstr}\n{img_chn_item_fr}\n}}"""
                        
            for key in model_dict.keys():
                if isinstance(model_dict[key], dict):
                    items = model_dict[key].items()
                    nested_list_in_dict = []
                        
                    for item in items:
                        if not isinstance(item[1], list):
                            key_val_pair_in_dict = [f"\n{item[0]}: {item[1]}\n" for item in items]
                            key_val_pair_in_dict_fr = "".join(key_val_pair_in_dict) 
                        key_val_pair_in_dict_output = f"""\n{key} {{\n {key_val_pair_in_dict_fr}\n}}"""
                                        
                        if isinstance(item[1], list):
                            for i in range(len(item[1])):
                                key_list_pair = f"\n{item[0]}: {item[1][i]}\n"
                                nested_list_in_dict.append(key_list_pair)
                            key_list_pair_fr = "".join(nested_list_in_dict)
                            key_list_pair_output = f"""\n{key} {{\n {key_list_pair_fr} \n}}"""  
                key_val_list_in_dict_fr = f"""{key_val_pair_in_dict_output} \n{key_list_pair_output}"""          
        
                model_set = f"""{config_type} {{{base_fr}\n{input_img_config}\n{key_val_list_in_dict_fr} \n}}"""
                print(model_set)
                return model_set

#%%  #########  training_config   ###############
def generate_training_config(json_file, nvidia_specfile_key: str ="nvidia_specfile",
                            config_type: str="training_config"
                            ):
    nv_file_json = json_file.get(nvidia_specfile_key)
    nv_file_keys = nv_file_json.keys()
    if config_type not in nv_file_keys:
        raise Exception(f"""The config_type parameter {config_type} is 
                        not a key in the specfile. Ensure that {config_type} 
                        is a key in the json file
                        """
                        )
    if isinstance(nv_file[config_type], dict):
        train_dict = nv_file_json[config_type]
        first_cl = [f"{key}: {train_dict[key]}\n" for key in train_dict.keys() 
                    if type(train_dict[key]) != dict
                    ]
        for key in train_dict.keys():
            if isinstance(train_dict[key], dict):
                n_dict = train_dict[key]
                for key_s in n_dict.keys():
                    if type(n_dict[key_s]) != dict:
                        if key_s in ["x", "y", "w", "h"]:
                            n_dict_items = train_dict[key].items()
                            sep_list = [f"""\n{key} {{\n"key": {n_dict_item[0]}\n"value": {n_dict_item[1]}\n}}"""#.strip()
                                        for n_dict_item in n_dict_items
                                        ]
                            sep_list_fr = "".join(sep_list)
                        else:
                            n_dict_items = train_dict[key].items()
                            all_items_n = [f"{n_dict_item[0]}: {n_dict_item[1]}\n" for n_dict_item in n_dict_items]
                            all_items_n_fr = "".join(all_items_n)
                            all_items_fn = f"\n{key} {{\n {all_items_n_fr} }}"
            
        each_item_nested = []
        more_nested_list = []
        for keys in train_dict.keys():
            if (type(train_dict[keys]))==dict: 
                sec_train_dict = train_dict[keys]
                for key_c in sec_train_dict.keys():
                    if type(sec_train_dict[key_c]) == dict:
                        sec_items = sec_train_dict[key_c].items()
                        key_c_list = []
                        for item in sec_items:
                            
                            if type(item[1]) != list:
                                nested_item= f"{item[0]}: {item[1]}\n"
                                key_c_list.append(nested_item)
                            elif type(item[1]) == list:
                                for i in range(len(item[1])):
                                    nnested_item = f"\n{item[0]}: {item[1][i]}\n"
                                    key_c_list.append(nnested_item)
                                    
                        format_each_item_nested = "".join(key_c_list)
                if keys not in ['classifier_regr_std', 'regularizer']:
                    more_nested = f"""\n{keys}{{\n{key_c}{{\n{format_each_item_nested}\n}}}}\n"""#.strip() 
                    more_nested_list.append(more_nested)      

                    more_nested_list_set = "".join(more_nested_list)
                        
            first_cl_set = "".join(first_cl)
        
        training_set = f"""training_config {{\n{first_cl_set}\n{sep_list_fr}\n{all_items_fn}\n{more_nested_list_set}\n}}"""
        print(training_set)
        return training_set


def generate_nvidia_specfile(json_file, 
                             nvidia_specfile_key: str ="nvidia_specfile"
                             ):
    nv_file_json = json_file.get(nvidia_specfile_key)
    main_var = [f"{key}: {nv_file[key]}\n" for key in nv_file_json.keys()
                if not isinstance(nv_file[key], dict)
                ]
    main_fr = "".join(main_var)
    model_config = generate_model_config(json_file=json_file)
    dataset_config = generate_dataset_config(json_file=json_file)
    training_config = generate_training_config(json_file=json_file)
    inference_config = get_unnested_setting(json_file=json_file, config_type="inference_config")
    evaluate_config = get_unnested_setting(json_file=json_file, config_type="evaluation_config")
    specfile = f"""{main_fr}\n{model_config}\n{dataset_config}\n{training_config}\n{inference_config}\n{evaluate_config}"""
    print(specfile)
    return specfile

#%%
generate_nvidia_specfile(json_file=data)
    
    
    
    
    
    
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
