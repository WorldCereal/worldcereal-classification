This document describes how the WorldCereal system is triggered in the different cereal and maize seasons. Definition of the seasons is as follows:
  - `tc-wintercereals` is the global "winter wheat" season which considers winter-planted wheat. This does not consider optional dormancy requirements nor wheat variety. Therefore, it is considered the main wheat season anywhere in the world.
  - `tc-maize-main` is the main "maize" season, although in some regions of the world (parts of the Northern Hemisphere) this season also covers spring-planted wheat ("spring wheat"). This season therefore targets maize globally, and additionally spring wheat in dedicated regions.
  - `tc-maize-second` is an optional second "maize" season occurring in some parts of the world. There will be no wheat mapping in this season.

The temporary crops layer, which is used as a mask for all crop type, active marker and irrigation products, is typically generated after the end of the LATEST season within a given reference year and AEZ. So production of the temporary crops layer can coincide with the tc-wintercereals, tc-maize-main and tc-maize-second season depending on the timing of eah of these seasons within the AEZ.

Based on these definitions, the triggers, periods considered, and detector models are described below.

## Season / Detector Mapping:

### Winter wheat triggers

| `tc-wintercereals` trigger |
| -------------------------- |

| Product                |                   Period                   |               Detector |                                                                                                                                                                                                          Model |                                                                             Remarks |
| ---------------------- | :----------------------------------------: | ---------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ----------------------------------------------------------------------------------: |
| Annual cropland        |     [season_end - 1 year, season_end]      |      Cropland detector |           [WorldCerealPixelCatBoost_cropland](https://artifactory.vgt.vito.be:443/auxdata-public/worldcereal/models/WorldCerealPixelCatBoost/v512/cropland_detector_WorldCerealPixelCatBoost_v512/config.json) | :exclamation: Only when this season is the LAST season to end in the reference year |
| Winter cereals         | [season_start - winter buffer, season_end] | Wintercereals detector | [WorldCerealPixelCatBoost_wintercereals](https://artifactory.vgt.vito.be:443/auxdata-public/worldcereal/models/WorldCerealPixelCatBoost/v502/wintercereals_detector_WorldCerealPixelCatBoost_v502/config.json) |                                                                                   - |
| Active cropland marker | [season_start - winter buffer, season_end] |        Season detector |                                                                                                                                                                                                           None |                                                                                   - |
| Irrigation             | [season_start - winter buffer, season_end] |    Irrigation detector |       [WorldCerealPixelCatBoost_irrigation](https://artifactory.vgt.vito.be:443/auxdata-public/worldcereal/models/WorldCerealPixelCatBoost/v420/irrigation_detector_WorldCerealPixelCatBoost_v420/config.json) |                                                                                   - |


### Maize triggers

| `summer1` trigger |
| ----------------- |

| Product                |                   Period                    |                Detector |                                                                                                                                                                                                          Model |                                                                             Remarks |
| ---------------------- | :-----------------------------------------: | ----------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ----------------------------------------------------------------------------------: |
| Annual cropland        |      [season_end - 1 year, season_end]      |       Cropland detector |           [WorldCerealPixelCatBoost_cropland](https://artifactory.vgt.vito.be:443/auxdata-public/worldcereal/models/WorldCerealPixelCatBoost/v512/cropland_detector_WorldCerealPixelCatBoost_v512/config.json) | :exclamation: Only when this season is the LAST season to end in the reference year |
| Maize                  | [season_start - summer1 buffer, season_end] |          Maize detector |                 [WorldCerealPixelCatBoost_maize](https://artifactory.vgt.vito.be:443/auxdata-public/worldcereal/models/WorldCerealPixelCatBoost/v502/maize_detector_WorldCerealPixelCatBoost_v502/config.json) |                                                                                   - |
| Spring cereals         | [season_start - summer1 buffer, season_end] | Spring cereals detector | [WorldCerealPixelCatBoost_springcereals](https://artifactory.vgt.vito.be:443/auxdata-public/worldcereal/models/WorldCerealPixelCatBoost/v502/springcereals_detector_WorldCerealPixelCatBoost_v502/config.json) |                                        :exclamation: Only when AEZ `trigger_sw = 1` |
| Active cropland marker | [season_start - summer1 buffer, season_end] |         Season detector |                                                                                                                                                                                                           None |                                                                                   - |
| Irrigation             | [season_start - summer1 buffer, season_end] |     Irrigation detector |       [WorldCerealPixelCatBoost_irrigation](https://artifactory.vgt.vito.be:443/auxdata-public/worldcereal/models/WorldCerealPixelCatBoost/v420/irrigation_detector_WorldCerealPixelCatBoost_v420/config.json) |                                                                                   - |


| `summer2` trigger |
| ----------------- |

| Product                |                   Period                    |            Detector |                                                                                                                                                                                                    Model |                                                                             Remarks |
| ---------------------- | :-----------------------------------------: | ------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ----------------------------------------------------------------------------------: |
| Annual cropland        |      [season_end - 1 year, season_end]      |   Cropland detector |     [WorldCerealPixelCatBoost_cropland](https://artifactory.vgt.vito.be:443/auxdata-public/worldcereal/models/WorldCerealPixelCatBoost/v512/cropland_detector_WorldCerealPixelCatBoost_v512/config.json) | :exclamation: Only when this season is the LAST season to end in the reference year |
| Maize                  | [season_start - summer2 buffer, season_end] |      Maize detector |           [WorldCerealPixelCatBoost_maize](https://artifactory.vgt.vito.be:443/auxdata-public/worldcereal/models/WorldCerealPixelCatBoost/v502/maize_detector_WorldCerealPixelCatBoost_v502/config.json) |                                                                                   - |
| Active cropland marker | [season_start - summer2 buffer, season_end] |     Season detector |                                                                                                                                                                                                     None |                                                                                   - |
| Irrigation             | [season_start - summer2 buffer, season_end] | Irrigation detector | [WorldCerealPixelCatBoost_irrigation](https://artifactory.vgt.vito.be:443/auxdata-public/worldcereal/models/WorldCerealPixelCatBoost/v420/irrigation_detector_WorldCerealPixelCatBoost_v420/config.json) |                                                                                   - |
 


## Required processing buffers
While the system takes into account the globally prescribed seasonality of wheat and maize, actual processing needs to include a buffer before the exact start of each season.
The purpose of the buffer is to capture a potential slight shift in the growing season of a particular year. Especially a backward shift would impact GDD normalization and detection of phenological features.

The default buffer (in days) prior to the prescribed season start is the amount of days that will be subtracted from the original season start and become the new season start as used by the system:
```
SEASON_BUFFER = {
    'winter': 15,
    'summer1': 15,
    'summer2': 15,
    'annual': 0,
    'custom': 0
}
```
