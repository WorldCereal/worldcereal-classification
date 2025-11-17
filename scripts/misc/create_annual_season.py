# %%

import numpy as np
import rasterio
from pathlib import Path

date = "20251030"
basedir = Path(
    f"/vitodata/worldcereal/data/cropcalendars/Phase_II/worldcereal_cropcalendars/_PUBLISHED/{date}"
)

s1_file = basedir / "S1_EOS_WGS84.tif"
s2_file = basedir / "S2_EOS_WGS84.tif"

s1_data = rasterio.open(s1_file).read(1).astype(int)
s2_data = rasterio.open(s2_file).read(1).astype(int)

assert s1_data.shape == s2_data.shape
print(s1_data.shape)
print(s2_data.shape)

# %%
print(np.unique(s2_data.astype(int)))
# annual_eos = np.zeros_like(m1_data)

# %%
annual_eos = np.maximum(s1_data, s2_data)
annual_sos = annual_eos - 364
annual_sos[annual_sos < 0] = annual_sos[annual_sos < 0] + 365
annual_sos[annual_eos == 0] = 0
print(annual_sos[100, 500])
print(annual_eos[100, 500])
np.unique(annual_eos)

# %%
with rasterio.open(s1_file) as src:
    kwargs = src.meta.copy()
kwargs.update({"compress": "deflate"})
print(kwargs)


# %%
annual_sos_file = basedir / "ANNUAL_SOS_WGS84.tif"
annual_eos_file = basedir / "ANNUAL_EOS_WGS84.tif"
annual_sos = annual_sos.astype(np.int16)
annual_eos = annual_eos.astype(np.int16)
with rasterio.open(annual_sos_file, "w", **kwargs) as dst:
    dst.write(annual_sos, 1)
with rasterio.open(annual_eos_file, "w", **kwargs) as dst:
    dst.write(annual_eos, 1)

# %%
