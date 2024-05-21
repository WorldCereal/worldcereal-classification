#%%

import xarray as xr
import matplotlib.pyplot as plt

output = xr.open_dataset('2024_05_17_13_41_40_input_cube_worldCereal.nc')
output = output['B08'].to_numpy().squeeze()[0,:,:].squeeze()
plt.imshow(output)

#%%

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

output = xr.open_dataset('2024_05_17_14_00_16_output_presto.nc')
output.drop_vars('crs')

flatten_output = output.to_array()

#flatten_output = flatten_output.flatten()
#plt.hist(flatten_output)
#plt.show()

#nan_counts = np.isnan(flatten_output).sum()/np.prod(flatten_output.shape)
