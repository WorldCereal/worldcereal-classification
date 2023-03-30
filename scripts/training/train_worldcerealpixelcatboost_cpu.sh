#!/usr/bin/env bash

export SPARK_HOME=/usr/hdp/current/spark2-client
export PATH="$SPARK_HOME/bin:$PATH"
export PYTHONPATH=wczip

cd src
zip -r ../dist/worldcereal.zip worldcereal
cd ..

PYSPARK_PYTHON=./ewocenv/bin/python \
${SPARK_HOME}/bin/spark-submit \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON="./ewocenv/bin/python" \
--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON="./ewocenv/bin/python" \
--conf spark.executorEnv.LD_LIBRARY_PATH="./ewocenv/lib" \
--conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH="./ewocenv/lib" \
--conf spark.executorEnv.PYSPARK_PYTHON="./ewocenv/bin/python" \
--conf spark.yarn.appMasterEnv.PYTHON_EGG_CACHE=./ \
--conf spark.executorEnv.GDAL_CACHEMAX=128 \
--conf spark.yarn.appMasterEnv.XDG_CACHE_HOME=.cache \
--conf spark.executorEnv.XDG_CACHE_HOME=.cache \
--conf spark.speculation=false \
--conf spark.executor.instances=10 \
--conf spark.driver.cores=16 \
--conf spark.executor.cores=16 \
--conf spark.task.cpus=16 \
--conf spark.sql.broadcastTimeout=500000 \
--conf spark.driver.memoryOverhead=16g --conf spark.executor.memoryOverhead=16g \
--conf spark.memory.fraction=0.2 \
--conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.am.waitTime=500s \
--conf spark.driver.maxResultSize=0 \
--master yarn --deploy-mode cluster --queue default \
--conf spark.app.name="worldcereal-trainmodels" \
--py-files /data/worldcereal/software/wheels/satio-1.1.16a1-py3-none-any.whl \
--archives "dist/worldcereal.zip#wczip","hdfs:///tapdata/worldcereal/worldcereal_gdal3.tar.gz#ewocenv" \
src/worldcereal/train/worldcerealpixelcatboost.py \
