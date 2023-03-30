#!/usr/bin/env bash

export SPARK_HOME=/usr/hdp/current/spark2-client
export PATH="$SPARK_HOME/bin:$PATH"
export PYTHONPATH=wczip

cd src
zip -r ../dist/worldcereal.zip worldcereal
cd ..

################
# SETTINGS
################

FILE='/data/worldcereal/cib/CIB_V1/POINT/2021_UKR_sunflowermap/2021_UKR_sunflowermap_POINT_110_samples.json'
CIB='CIB_V1'

DRIVERMEM='6g'
EXMEM='1g'

################

PYSPARK_PYTHON=./ewocenv/bin/python \
${SPARK_HOME}/bin/spark-submit \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON="./ewocenv/bin/python" \
--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON="./ewocenv/bin/python" \
--conf spark.executorEnv.LD_LIBRARY_PATH="./ewocenv/lib" \
--conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH="./ewocenv/lib" \
--conf spark.executorEnv.PYSPARK_PYTHON="./ewocenv/bin/python" \
--conf spark.yarn.am.waitTime=500s \
--executor-memory 4g --driver-memory 4g \
--conf spark.executorEnv.GDAL_CACHEMAX=512 \
--conf spark.driver.memoryOverhead=${DRIVERMEM} --conf spark.executor.memoryOverhead=${EXMEM} \
--conf spark.memory.fraction=0.2 \
--conf spark.shuffle.service.enabled=true --conf spark.dynamicAllocation.enabled=true \
--conf spark.dynamicAllocation.maxExecutors=400 \
--master yarn --deploy-mode cluster --queue default \
--conf spark.speculation=true \
--conf spark.app.name="CIB-INPUTS-FROM-EWOC" \
--py-files /data/worldcereal/software/wheels/satio-1.1.11-py3-none-any.whl \
--archives "dist/worldcereal.zip#wczip","hdfs:///tapdata/worldcereal/worldcereal_gdal3.tar.gz#ewocenv" \
scripts/cib/inputs/inputs_from_ewocdata.py \
--file ${FILE} \
--cib ${CIB} \
-s
