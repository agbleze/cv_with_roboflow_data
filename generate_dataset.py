
#%%
from datumaro.plugins.synthetic_data import FractalImageGenerator



# %%
# link for model download https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/
fragen = FractalImageGenerator(output_dir="gendata", count=100, shape=(700,700)).generate_dataset()
# %%
fragen.generate_dataset()
# %%
