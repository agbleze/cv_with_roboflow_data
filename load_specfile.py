

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
    nest_cl = [f"""classifier_regr_std {{
               "key": {item[0]}
               "value": {item[1]}
               }}
               """
               for item in train_dict["classifier_regr_std"].items()

               
                #for key in train_dict.keys() 
                #if (type(train_dict[key]) == dict) and (train_dict.keys() == "classifier_regr_std")
                ]
    each_item_nested = []
    d_item_nested = []
    more_nested_list = []
    for keys in train_dict.keys():
        if (type(train_dict[keys]))==dict: 
            sec_train_dict = train_dict[keys]
            for key_c in sec_train_dict.keys():
                if type(sec_train_dict[key_c]) == dict:
                    sec_items = sec_train_dict[key_c].items()
                    print(sec_items)
                    for item in sec_items:
                        
                        if type(item[1]) != list:
                            nested_item= f"{item[0]}: {item[1]}\n"#.strip()
                            each_item_nested.append(nested_item)
                        elif type(item[1]) == list:
                            
                            for i in range(len(item[1])):
                                nnested_item = f"{item[0]}: {item[1][i]}\n"
                                each_item_nested.append(nnested_item)
                                
            format_each_item_nested = "".join(each_item_nested)#.strip()
            more_nested = f"""{keys}{{\n
                {key_c}{{\n
                {format_each_item_nested}
                }}
            }}\n"""#.strip() 
            more_nested_list.append(more_nested)      
    
    more_nested_list_set = "".join(more_nested_list)
                    
            
    nest_set = "".join(nest_cl)#.strip()
    
    training_set = f"""training_config {{\n
                    {"".join(first_cl)}
                    {nest_set}
                    {more_nested_list_set}
                    }}
                    """
    print(training_set)
    #print(nest_set)


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
