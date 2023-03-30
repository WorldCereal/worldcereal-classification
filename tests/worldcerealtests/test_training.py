import tempfile
from worldcereal.utils.training import get_pixel_data
from worldcereal.train.worldcerealpixelcatboost import Trainer
from worldcereal.train import get_training_settings


def test_get_pixel_data(training_df_LC):

    detector = 'cropland'
    aez_zoneid = 46172

    # Get the trainingsettings
    trainingsettings = get_training_settings(detector)

    # Replace training files with test resources
    trainingsettings['cal_df_files'] = [training_df_LC]
    trainingsettings['val_df_files'] = [training_df_LC]
    trainingsettings['test_df_files'] = [training_df_LC]

    df = get_pixel_data(detector, trainingsettings,
                        trainingsettings['bands'],
                        outlierinputs=trainingsettings['outlierinputs'],
                        aez_zone=aez_zoneid,
                        buffer=500000,
                        scale_features=False,
                        minsamples=10,
                        outlierfraction=0.1)

    assert df is not None


def test_training_catboost(training_df_LC):

    detector = 'cropland'

    # Create tmp directory and tmp model file in it
    basedir = tempfile.TemporaryDirectory().name

    # Get the trainingsettings
    trainingsettings = get_training_settings(detector)

    # replace minsamples
    trainingsettings['minsamples'] = [25, 25, 25]

    # Replace training files with test resources
    trainingsettings['cal_df_files'] = [str(training_df_LC)]
    trainingsettings['val_df_files'] = [str(training_df_LC)]
    trainingsettings['test_df_files'] = [str(training_df_LC)]

    # Initialize trainer
    trainer = Trainer(trainingsettings, basedir, detector)

    # First train the base model
    trainer.train_base()
    trainer.train_groups([46000], outlierfraction=0.10,
                         train_from_scratch=True)
