
#%%
import json
from specfile_parser import generate_nvidia_specfile


specfile = "/home/lin/cv_with_roboflow_data/nvidia_tao_tut/default_specfile.json"

with open(specfile, "r") as spec:
    data = json.loads(spec.read())

#%%    
file = generate_nvidia_specfile(json_file=data)

# %%
with open("default_experiment_specfile.cfg", "w") as f:
    f.writelines(file)

# %%
