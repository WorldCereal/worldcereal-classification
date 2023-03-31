import os

from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
from tensorflow.keras.models import Model


def fix_pickle():

    def unpack(model, training_config, weights):
        restored_model = deserialize(model)
        if training_config is not None:
            restored_model.compile(
                **saving_utils.compile_args_from_training_config(
                    training_config
                )
            )
        restored_model.set_weights(weights)
        return restored_model

    # Hotfix function
    def make_keras_picklable():

        def __reduce__(self):
            model_metadata = saving_utils.model_metadata(self)
            training_config = model_metadata.get("training_config", None)
            model = serialize(self)
            weights = self.get_weights()
            return (unpack, (model, training_config, weights))

        cls = Model
        cls.__reduce__ = __reduce__

    # Run the function
    make_keras_picklable()


def get_spark_context(name="WORLDCEREAL", localspark=False,
                      pythonpath='/data/users/Public/kristofvt/'
                      'python/worldcereal/bin/python'):

    import pyspark.serializers
    import cloudpickle
    import pyspark
    from pyspark.sql import SparkSession

    pyspark.serializers.cloudpickle = cloudpickle

    if not localspark:

        # Hot fix to make keras models pickable
        fix_pickle()

        spark = SparkSession.builder \
            .appName(name) \
            .getOrCreate()
        sc = spark.sparkContext
    else:
        os.environ['PYSPARK_PYTHON'] = pythonpath
        spark = SparkSession.builder \
            .appName(name) \
            .master('local[1]') \
            .config('spark.driver.host', '127.0.0.1') \
            .config('spark.executor.memory', '2G') \
            .config('spark.driver.memory', '2G') \
            .getOrCreate()
        sc = spark.sparkContext

    # Set log level
    sc.setLogLevel("WARN")

    return sc
