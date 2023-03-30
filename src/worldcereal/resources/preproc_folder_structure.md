<h3> Folder structure </h3>

The proposed WorldCereal preprocessed products folder structure should be the following:

```
WORLDCEREAL_PREPROC/
├── OPTICAL
│   └── <tile[:2]>
│       └── <tile[2]>
│           └── <tile[3:]>
│               └── <year>
│                   └── <yyyymmdd>
│                       └── <platform>_<processing_level>_<YYYYMMDDTHHMMSS>_<unique_id>_<s2_tile_id>
│                           └── <platform>_<atcor_algo>_<YYYYMMDDTHHMMSS>_<unique_id>_<s2_tile_id>_<band>.tif
├── SAR
│   └── <tile[:2]>
│       └── <tile[2]>
│           └── <tile[3:]>
│               └── <year>
│                   └── <yyyymmdd>
│                       └── S1<platform_letter>_<YYYYMMDDTHHMMSS>_<orbit_direction>_<relative_orbit>_<unique_id>_<s2_tile_id>
│                           └── S1<platform_letter>_<YYYYMMDDTHHMMSS>_<orbit_direction>_<relative_orbit>_<unique_id>_<s2_tile_id>_<variable>_<band>.tif
└── TIR
    └── <tile[:2]>
        └── <tile[2]>
            └── <tile[3:]>
                └── <year>
                    └── <yyyymmdd>
                        └── LC08_L2SP_<YYYYMMDDTHHMMSS>_<unique_id>_<s2_tile_id>
                            └── LC08_L2SP_<YYYYMMDDTHHMMSS>_<unique_id>_<s2_tile_id>_<band>.tif
```

<h3> OPTICAL products </h3>

|Sentinel-2<sup>1</sup>|Landsat 8|WorldCereal|Resolution|
--- | --- | --- | ---
|B02|B02|B02|10m|
|B03|B03|B03|10m|
|B04|B04|B04|10m|
|B05|-|B05|20m|
|B06|-|B06|20m|
|B07|-|B07|20m|
|B08|B05<sup>2</sup>|B08|10m|
|B11|B06|B11|20m|
|B12|B07|B12|20m|
|SCL|FMASK|MASK|20m|

<sup>1</sup>Note that the bands not listed in this table are currently not used by the WorldCereal system.

<sup>2</sup>Note that according to Sen2Like convention, B05 should be matched with B8A. However, we opt to match it with B08, understanding that the calibration will be not as good, but that the 10m resolution of Sentinel-2 B08 is more important.

**Product naming**

* platform = S2A/S2B/LC08/LO08
* processing_level = MSIL1C/L1T
* unique_id = <[S2 product discriminator](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/naming-convention)> (Sentinel-2) / ```<WRS path><WRS row><YYYYMMDD><s2_tile_id>``` (Landsat 8)
* atcor_algo = SMAC/SCL/FMASK/...
* band = B02/B03/B04/B05/B06/B07/B08/B81/B11/B12/MASK


<h3> SAR products </h3>

|Sentinel-1|Resolution|
--- | --- |
|VV|20m|
|VH|20m|

**Product naming**

* platform_letter = A/B
* orbit_direction = ASC/DES
* unique_id = ```<absolute orbit number><mission data take ID><product unique ID>``` (e.g. 03415403F79AED83)
* variable = SIGMA0
* band = VV/VH


<h3> TIF options required </h3>

For **10m** output files:

```
blockxsize=1024
blockysize=1024
compress=deflate
dtype=uint16
tiled=True
nodata=0
```
For **20m** output files:

```
blockxsize=512
blockysize=512
compress=deflate
dtype=uint16
tiled=True
nodata=0
```

