
#%%
from datumaro.plugins.synthetic_data import FractalImageGenerator



# %%
fragen = FractalImageGenerator(output_dir="gendata", count=100, shape=(700,700)).generate_dataset()
# %%
fragen.generate_dataset()
# %%
