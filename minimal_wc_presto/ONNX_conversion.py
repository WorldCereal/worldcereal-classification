#%% Catboost
import catboost
from catboost.utils import convert_to_onnx_object
import onnx

# Load your CatBoost model
model = catboost.CatBoost()
model.load_model('./model/catboost.cbm')

onnx_model = convert_to_onnx_object(model)
onnx.save(onnx_model, './model/wc_catboost.onnx')





#%% For the pytorch model we need to know the input shape

import torch
from presto.presto import Presto
from model_class import PrestoFeatureExtractor
import xarray as xr
import numpy as np

#load the data
ds = xr.open_dataset("./data/belgium_good_2020-12-01_2021-11-30.nc", engine='netcdf4')
arr = ds.drop('crs').to_array(dim='bands')


# Load the Presto model
PRESTO_PATH = './model/presto.pt'
presto_model = Presto.load_pretrained(model_path=PRESTO_PATH, strict=False)
presto_extractor = PrestoFeatureExtractor(presto_model)

#get the required presto input through the feature extractor
input = presto_extractor.create_presto_input(arr)

x_sample = torch.tensor(np.expand_dims(input[0][0], axis=0), dtype=torch.float32)   # Shape matches the shape of eo data in your DataLoader
dw_sample = torch.tensor(np.expand_dims(input[1][0], axis=0), dtype=torch.long)     # Shape matches the shape of dynamic_world data in your DataLoader
month_sample = torch.tensor(np.expand_dims(input[2][0], axis = 0), dtype=torch.long)  # Shape matches the shape of months data in your DataLoader
latlons_sample = torch.tensor(np.expand_dims(input[3][0], axis = 0), dtype=torch.float32)  # Shape matches the shape of latlons data in your DataLoader
mask_sample = torch.tensor(np.expand_dims(input[4][0], axis = 0), dtype=torch.int)  

encoder_model = presto_model.encoder



with torch.no_grad():
    encoder_output = encoder_model(
        x_sample,  # Add batch dimension
        dynamic_world=dw_sample,  # Add batch dimension
        mask=mask_sample,  # Add batch dimension
        latlons=latlons_sample,  # Add batch dimension
        month=month_sample  # Add batch dimension
    )

    #%%

# Export the encoder model to ONNX
torch.onnx.export(
    encoder_model,
    (x_sample, dw_sample, latlons_sample,mask_sample, month_sample),
    './model/wc_presto.onnx',
    input_names=["x", "dynamic_world", "latlons", "mask", "month"],
    output_names=["output"],
    dynamic_axes={
        "x": {0: "batch_size"},
        "dynamic_world": {0: "batch_size"},
        "mask": {0: "batch_size"},
        "latlons": {0: "batch_size"},
        "month": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)
#%%
# Export the model to ONNX
torch.onnx.export(
    encoder_model,
    (x_sample, dw_sample, latlons_sample, month_sample, mask_sample),
    './model/wc_presto.onnx',
    input_names=["x", "dynamic_world", "latlons", "month", "mask"],
    output_names=["output"],
    dynamic_axes={
        "x": {0: "batch_size"},
        "dynamic_world": {0: "batch_size"},
        "mask": {0: "batch_size"},
        "latlons": {0: "batch_size"},
        "month": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)