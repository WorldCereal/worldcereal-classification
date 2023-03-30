#!/usr/bin/env bash

export SPARK_HOME=/usr/hdp/current/spark2-client
export PATH="$SPARK_HOME/bin:$PATH"
export PYTHONPATH=wczip

cd src
zip -r ../dist/worldcereal.zip worldcereal
cd ..

DRIVERMEM='20g'
DRIVERJAVAMEM='20g'
EXMEM='20g'
EXJAVAMEM='20g'

# DRIVERMEM='4g'
# DRIVERJAVAMEM='2g'
# EXMEM='1g'
# EXJAVAMEM='1g'

################

PYSPARK_PYTHON=./ewocenv/bin/python \
${SPARK_HOME}/bin/spark-submit \
--num-executors 10 \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON="./ewocenv/bin/python" \
--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON="./ewocenv/bin/python" \
--conf spark.executorEnv.LD_LIBRARY_PATH="./ewocenv/lib" \
--conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH="./ewocenv/lib" \
--conf spark.executorEnv.PYSPARK_PYTHON="./ewocenv/share/proj" \
--conf spark.yarn.appMasterEnv.PROJ_LIB="./ewocenv/share/proj" \
--conf spark.executorEnv.PROJ_LIB="./ewocenv/bin/python" \
--conf spark.yarn.appMasterEnv.XDG_CACHE_HOME=.cache \
--conf spark.executorEnv.XDG_CACHE_HOME=.cache \
--conf spark.yarn.am.waitTime=500s \
--executor-memory ${EXJAVAMEM} --driver-memory ${DRIVERJAVAMEM} \
--conf spark.executorEnv.GDAL_CACHEMAX=512 \
--conf spark.driver.memoryOverhead=${DRIVERMEM} --conf spark.executor.memoryOverhead=${EXMEM} \
--conf spark.memory.fraction=0.2 \
--conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.enabled=false \
--conf spark.dynamicAllocation.minExecutors=15 \
--conf spark.dynamicAllocation.maxExecutors=15 \
--master yarn --deploy-mode cluster --queue default \
--conf spark.hadoop.fs.permissions.umask-mode=022 \
--conf spark.speculation=false \
--conf spark.app.name="CIB-PREPROCESSTRAININGPOINTS" \
--py-files /data/worldcereal/software/wheels/satio-1.1.11-py3-none-any.whl \
--archives "dist/worldcereal.zip#wczip","hdfs:///tapdata/worldcereal/worldcereal_gdal3.tar.gz#ewocenv" \
scripts/cib/labels/preprocess_trainingpoints.py