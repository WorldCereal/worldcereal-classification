#!/usr/bin/env bash

export SPARK_HOME=/opt/spark3_2_0/
export PATH="$SPARK_HOME/bin:$PATH"
export PYTHONPATH=wczip

cd src || exit
zip -r ../dist/worldcereal.zip worldcereal
cd ..

EX_JAVAMEM='2g'
EX_PYTHONMEM='14g'
DR_JAVAMEM='8g'
DR_PYTHONMEM='16g'

PYSPARK_PYTHON=./ewocenv/bin/python \
${SPARK_HOME}/bin/spark-submit \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON="./ewocenv/bin/python" \
--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON="./ewocenv/bin/python" \
--conf spark.executorEnv.LD_LIBRARY_PATH="./ewocenv/lib" \
--conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH="./ewocenv/lib" \
--conf spark.executorEnv.PYSPARK_PYTHON="./ewocenv/bin/python" \
--conf spark.yarn.appMasterEnv.PYTHONPATH=$PYTHONPATH \
--conf spark.executorEnv.PYTHONPATH=$PYTHONPATH \
--executor-memory ${EX_JAVAMEM} --driver-memory ${DR_JAVAMEM} \
--conf spark.yarn.appMasterEnv.PYTHON_EGG_CACHE=./ \
--conf spark.executorEnv.GDAL_CACHEMAX=512 \
--conf spark.speculation=true \
--conf spark.sql.broadcastTimeout=36000 \
--conf spark.shuffle.registration.timeout=36000 \
--conf spark.sql.execution.arrow.pyspark.enabled=true \
--conf spark.shuffle.memoryFraction=0.6  \
--conf spark.hadoop.mapreduce.output.fileoutputformat.compress=true \
--conf spark.hadoop.mapreduce.output.fileoutputformat.compress.codec=org.apache.hadoop.io.compress.GzipCodec \
--conf spark.driver.memoryOverhead=${DR_PYTHONMEM} --conf spark.executor.memoryOverhead=${EX_PYTHONMEM} \
--conf spark.memory.fraction=0.2 \
--conf spark.executor.cores=4 \
--conf spark.task.cpus=4 \
--conf spark.driver.cores=4 \
--conf spark.dynamicAllocation.maxExecutors=1000 \
--conf spark.shuffle.service.enabled=true --conf spark.dynamicAllocation.enabled=true \
--conf spark.driver.maxResultSize=0 \
--master yarn --deploy-mode cluster --queue default \
--py-files "/vitodata/worldcereal/software/wheels/presto_worldcereal-0.1.5-py3-none-any.whl" \
--archives "dist/worldcereal.zip#wczip", "hdfs:///tapdata/worldcereal/worldcereal.tar.gz#ewocenv" \
--conf spark.app.name="worldcereal-presto_features" \
scripts/spark/compute_presto_features.py \
