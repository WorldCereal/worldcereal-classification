import os
import sys

from loguru import logger


def get_spark_context(
    name="WORLDCEREAL", localspark=False, threads="*", spark_version="3_2_0"
):
    """
    Returns SparkContext for local run.
    if local is True, conf is ignored.

    Customized for VITO MEP
    """
    if localspark:
        SPARK_HOME_2_0_0 = "/usr/hdp/current/spark2-client"
        SPARK_HOME_3_0_0 = "/opt/spark3_0_0"
        SPARK_HOME_3_2_0 = "/opt/spark3_2_0"

        SPARK_HOME = {
            "2_0_0": SPARK_HOME_2_0_0,
            "3_0_0": SPARK_HOME_3_0_0,
            "3_2_0": SPARK_HOME_3_2_0,
        }

        PY4J = {
            "2_0_0": "py4j-0.10.7",
            "3_0_0": "py4j-0.10.8.1",
            "3_2_0": "py4j-0.10.9.2",
        }

        SPARK_MAJOR_VERSION = {"2_0_0": "2", "3_0_0": "3", "3_2_0": "3"}

        spark_home = SPARK_HOME[spark_version]
        py4j_version = PY4J[spark_version]
        spark_major_version = SPARK_MAJOR_VERSION[spark_version]

        spark_py_path = [
            f"{spark_home}/python",
            f"{spark_home}/python/lib/{py4j_version}-src.zip",
        ]

        env_vars = {
            "SPARK_MAJOR_VERSION": spark_major_version,
            "SPARK_HOME": spark_home,
        }
        for k, v in env_vars.items():
            logger.info(f"Setting env var: {k}={v}")
            os.environ[k] = v

        logger.info(f"Prepending {spark_py_path} to PYTHONPATH")
        sys.path = spark_py_path + sys.path

        import py4j

        logger.info(f"py4j: {py4j.__file__}")

        import pyspark

        logger.info(f"pyspark: {pyspark.__file__}")

        import cloudpickle
        import pyspark.serializers
        from pyspark import SparkConf, SparkContext

        pyspark.serializers.cloudpickle = cloudpickle

        logger.info(f"Setting env var: PYSPARK_PYTHON={sys.executable}")
        os.environ["PYSPARK_PYTHON"] = sys.executable

        conf = SparkConf()
        conf.setMaster(f"local[{threads}]")
        conf.set("spark.driver.bindAddress", "127.0.0.1")

        sc = SparkContext(conf=conf)

    else:
        import cloudpickle
        import pyspark.serializers
        from pyspark.sql import SparkSession

        pyspark.serializers.cloudpickle = cloudpickle

        spark = SparkSession.builder.appName(name).getOrCreate()
        sc = spark.sparkContext

    return sc
