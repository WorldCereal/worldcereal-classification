#!/usr/bin/env bash
# shellcheck disable=SC2140

export SPARK_HOME=/opt/spark3_2_0/
export PATH="$SPARK_HOME/bin:$PATH"
export PYTHONPATH=wczip

cd src || exit
zip -r ../dist/worldcereal.zip worldcereal
cd ..

EX_JAVAMEM='8g'
EX_PYTHONMEM='16g'
DR_JAVAMEM='8g'
DR_PYTHONMEM='16g'

PYSPARK_PYTHON=./ewocenv/bin/python \
${SPARK_HOME}/bin/spark-submit \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON="./ewocenv/bin/python" \
--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON="./ewocenv/bin/python" \
--conf spark.executorEnv.LD_LIBRARY_PATH="./ewocenv/lib" \
--conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH="./ewocenv/lib" \
--conf spark.yarn.appMasterEnv.PYTHONPATH=$PYTHONPATH \
--conf spark.executorEnv.PYTHONPATH=$PYTHONPATH \
--executor-memory ${EX_JAVAMEM} --driver-memory ${DR_JAVAMEM} \
--conf spark.yarn.appMasterEnv.PYTHON_EGG_CACHE=./ \
--conf spark.executorEnv.GDAL_CACHEMAX=128 \
--conf spark.yarn.appMasterEnv.XDG_CACHE_HOME=.cache \
--conf spark.executorEnv.XDG_CACHE_HOME=.cache \
--conf spark.rpc.message.maxSize=1024 \
--conf spark.speculation=false \
--conf spark.executor.instances=1 \
--conf spark.driver.cores=16 \
--conf spark.executor.cores=16 \
--conf spark.task.cpus=16 \
--conf spark.sql.broadcastTimeout=500000 \
--conf spark.driver.memoryOverhead=${DR_PYTHONMEM} --conf spark.executor.memoryOverhead=${EX_PYTHONMEM} \
--conf spark.memory.fraction=0.2 \
--conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.am.waitTime=500s \
--conf spark.driver.maxResultSize=0 \
--master yarn --deploy-mode cluster --queue default \
--conf spark.app.name="worldcereal-trainmodels" \
--archives "dist/worldcereal.zip#wczip","hdfs:///tapdata/worldcereal/worldcereal_python38.tar.gz#ewocenv" \
scripts/spark/train_catboost.py \
