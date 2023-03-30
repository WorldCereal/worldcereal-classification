import tempfile
import json
from pathlib import Path

from worldcereal.classification.models import WorldCerealModel
from worldcereal.classification.models import WorldCerealFFNN
from worldcereal.classification.models import WorldCerealRFModel
from worldcereal.classification.models import WorldCerealCNN
from worldcereal.classification.models import WorldCerealPixelLSTM
from worldcereal.classification.models import WorldCerealPatchLSTM
from worldcereal.classification.models import WorldCerealCatBoostModel
from worldcereal.utils import get_best_model


import numpy as np
from tensorflow.keras.utils import to_categorical


def test_WorldCerealRFModel():

    # Create tmp directory and tmp model file in it
    basedir = tempfile.TemporaryDirectory().name
    modelfile = Path(basedir) / 'savedmodel'

    # Dummy feature names
    ft_names = ['ft1', 'ft2', 'ft3']

    # Test some RF parameters
    rf_parameters = {'n_estimators': 100,
                     'max_depth': 4,
                     'verbose': 1}

    # Setup a random forest model
    model = WorldCerealRFModel(feature_names=ft_names,
                               basedir=basedir,
                               parameters=rf_parameters)

    # Save the model
    model.save(modelfile)

    # Get the saved config file
    configfile = Path(basedir) / 'config.json'

    # Restore the model
    restoredmodel = WorldCerealRFModel.from_config(configfile)

    # Check if configurations are identical
    assert model.config == restoredmodel.config


def test_WorldCerealCatBoostModel():

    # Create tmp directory and tmp model file in it
    basedir = tempfile.TemporaryDirectory().name
    modelfile = Path(basedir) / 'savedmodel'

    # Dummy feature names
    ft_names = ['ft1', 'ft2', 'ft3']

    # Test some RF parameters
    rf_parameters = {'n_estimators': 100,
                     'max_depth': 4,
                     'verbose': 1}

    # Setup a random forest model
    model = WorldCerealCatBoostModel(feature_names=ft_names,
                                     basedir=basedir,
                                     parameters=rf_parameters)

    # Train on dummy data
    samples = 100
    inputdata = np.random.rand(samples, len(ft_names))
    outputdata = np.zeros((samples, ))
    outputdata[:int(samples/2)] = 1

    model.train(
        inputs=inputdata,
        outputs=outputdata,
    )

    # Save the model
    model.save(modelfile)

    # Get the saved config file
    configfile = Path(basedir) / 'config.json'

    # Restore the model
    restoredmodel = WorldCerealCatBoostModel.from_config(configfile)

    # Check if configurations are identical
    assert model.config == restoredmodel.config


def test_load_CNN():
    import worldcereal
    configfile = Path(worldcereal.__file__).parent / \
        'resources' / 'CNN_models' / \
        'WorldCerealPatchLSTM_cropland' / 'parentmodel' / 'config.json'
    _ = WorldCerealCNN.from_config(configfile)


def test_load_model_from_url():
    import worldcereal
    configfile = Path(worldcereal.__file__).parent / \
        'resources' / 'exampleconfigs' / \
        'example_bucketrun_annual_config.json'
    modelconfig = json.load(open(configfile, 'r'))['models'][
        'annualcropland']
    modelconfig = get_best_model(modelconfig, aez_id=12048,
                                 realm_id=5, use_local_models=True)

    _ = WorldCerealModel.from_config(modelconfig)


def test_load_local_model_from_url():
    configfile = 'https://artifactory.vgt.vito.be:443/auxdata-public/worldcereal/models/WorldCerealPixelCatBoost/v720/wintercereals_detector_WorldCerealPixelCatBoost_v720/config.json'  # NOQA
    modelconfig = get_best_model(configfile, aez_id=25147,
                                 use_local_models=True)

    _ = WorldCerealModel.from_config(modelconfig)


# def test_load_realm_model_from_url():
#     import worldcereal
#     configfile = Path(worldcereal.__file__).parent / \
#         'resources' / 'exampleconfigs' / \
#         'example_bucketrun_annual_config.json'
#     modelconfig = json.load(open(configfile, 'r'))['models'][
#         'annualcropland']
#     modelconfig = get_best_model(modelconfig, aez_id=12048,
#                                  realm_id=5, use_local_models=True)

#     _ = WorldCerealModel.from_config(modelconfig)


def test_load_irr_model_from_url():
    import worldcereal
    configfile = Path(worldcereal.__file__).parent / \
        'resources' / 'exampleconfigs' / \
        'example_bucketrun_winter_config.json'
    modelconfig = json.load(open(configfile, 'r'))['parameters'][
        'irrmodels']['irrigation']
    _ = WorldCerealModel.from_config(modelconfig)


def test_WorldCerealCNN():

    # Create tmp directory and tmp model file in it
    basedir = tempfile.TemporaryDirectory().name
    modelfile = Path(basedir) / 'savedmodel'

    # Dummy feature names
    ft_names = ['DEM', 'biome01', 'biome02']

    # Windowsize is compulsary parameter
    unet_parameters = {'windowsize': 64}

    # Setup a UNET model
    model = WorldCerealCNN(feature_names=ft_names,
                           basedir=basedir,
                           parameters=unet_parameters)

    # Save the model
    model.save(modelfile)

    # Get the saved config file
    configfile = Path(basedir) / 'config.json'

    # Restore the model
    restoredmodel = WorldCerealCNN.from_config(configfile)

    # Check if configurations are identical
    assert model.config == restoredmodel.config


def test_WorldCerealFFNN():

    # Create tmp directory and tmp model file in it
    basedir = tempfile.TemporaryDirectory().name
    modelfile = Path(basedir) / 'savedmodel'

    # Dummy feature names
    ft_names = list(np.arange(32).astype(str))

    # FFNN parameters
    ffnn_parameters = {
        'nodes': 64,
        'depth': 3,
    }

    # Initialize model
    ffnnmodel = WorldCerealFFNN(
        feature_names=ft_names,
        basedir=basedir,
        parameters=ffnn_parameters)

    # Train on dummy data
    samples = 100
    inputdata = np.random.rand(samples, len(ft_names))
    outputdata = np.zeros((samples, ))
    outputdata[:int(samples/2)] = 1
    outputdata = to_categorical(outputdata, num_classes=2)
    ffnnmodel.train(inputdata, outputdata)

    # Save the model
    ffnnmodel.save(modelfile)

    # Get the saved config file
    configfile = Path(basedir) / 'config.json'

    # Restore the model
    restoredmodel = WorldCerealFFNN.from_config(configfile)

    # Check if configurations are identical
    assert ffnnmodel.config == restoredmodel.config


def test_transfer_WorldCerealCNN():

    # Create tmp directory and tmp model file in it
    basedir = tempfile.TemporaryDirectory().name
    modelfile = Path(basedir) / 'savedmodel'

    # Dummy feature names
    ft_names = ['DEM', 'biome01', 'biome02']

    # Create dummy data
    inputs = np.random.rand(100, 64*64, len(ft_names))
    outputs = np.zeros((100,))

    # Windowsize is compulsary parameter
    cnn_parameters = {'windowsize': 64}

    # Setup a CNN model
    model = WorldCerealCNN(feature_names=ft_names,
                           basedir=basedir,
                           parameters=cnn_parameters)
    model.save(modelfile)
    model.train(calibrationx=inputs,
                calibrationy=outputs,
                epochs=1)

    # Transfer the model
    newbasedir = tempfile.TemporaryDirectory().name
    transfermodelfile = Path(newbasedir) / 'savedmodel'
    transferredmodel = model.transfer(newbasedir)

    # Retrain the model on the last layer
    transferredmodel.retrain(calibrationx=inputs,
                             calibrationy=outputs,
                             epochs=1,
                             learning_rate=1e-6)

    # Save the retrained model
    transferredmodel.save(transfermodelfile)


def test_WorldCerealPixelLSTM():

    # Create tmp directory and tmp model file in it
    basedir = tempfile.TemporaryDirectory().name
    modelfile = Path(basedir) / 'savedmodel'

    # Feature names
    s1names = [f'SIGMA0-VV-ts{x}-20m' for x in range(18)]
    s2names = [f'L2A-B08-ts{x}-10m' for x in range(18)]
    precipnames = ['AgERA5-precipitation_flux-summation-20m']
    demnames = ['DEM-alt-10m', 'DEM-slo-10m']
    ft_names = s1names + s2names + precipnames + demnames

    # Initialize model
    rnnmodel = WorldCerealPixelLSTM(
        feature_names=ft_names,
        basedir=basedir
    )

    # Train on dummy data
    samples = 100
    inputdata = np.random.rand(samples, len(ft_names))
    outputdata = np.zeros((samples, ))
    outputdata[:int(samples/2)] = 1
    rnnmodel.train(inputdata, outputdata)

    # Save the model
    rnnmodel.save(modelfile)

    # Transfer the model
    newbasedir = tempfile.TemporaryDirectory().name
    transfermodelfile = Path(newbasedir) / 'savedmodel'
    transferredmodel = rnnmodel.transfer(newbasedir)

    # Retrain the model on the last layer
    transferredmodel.retrain(calibrationx=inputdata,
                             calibrationy=outputdata,
                             epochs=1,
                             learning_rate=1e-6)

    # Save the retrained model
    transferredmodel.save(transfermodelfile)


def test_WorldCerealPatchLSTM():

    # Create tmp directory and tmp model file in it
    basedir = tempfile.TemporaryDirectory().name
    modelfile = Path(basedir) / 'savedmodel'

    # Feature names
    s1names = [f'SAR-VV-ts{x}-20m' for x in range(15)]
    s2names = [f'OPTICAL-B08-ts{x}-10m' for x in range(18)]
    demnames = ['DEM-alt-10m', 'DEM-slo-10m']
    ft_names = s1names + s2names + demnames

    # Windowsize is compulsary parameter
    cnn_parameters = {'windowsize': 64}

    # Initialize model
    cnnmodel = WorldCerealPatchLSTM(
        feature_names=ft_names,
        basedir=basedir,
        parameters=cnn_parameters
    )

    # Train on dummy data
    samples = 10
    inputdata = np.random.rand(samples, 64*64, len(ft_names))
    outputdata = np.zeros((samples, ))
    outputdata[:int(samples/2)] = 1
    cnnmodel.train(inputdata, outputdata, epochs=1)

    # Save the model
    cnnmodel.save(modelfile)

    # Transfer the model
    newbasedir = tempfile.TemporaryDirectory().name
    transfermodelfile = Path(newbasedir) / 'savedmodel'
    transferredmodel = cnnmodel.transfer(newbasedir)

    # Retrain the model on the last layer
    transferredmodel.retrain(calibrationx=inputdata,
                             calibrationy=outputdata,
                             epochs=1,
                             learning_rate=1e-6)

    # Save the retrained model
    transferredmodel.save(transfermodelfile)
