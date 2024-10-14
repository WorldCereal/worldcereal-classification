import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool
from loguru import logger
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from worldcereal.train import get_training_data

# from worldcereal.classification.models import WorldCerealCatBoostModel
# from worldcereal.classification.weights import get_refid_weight
from worldcereal.utils.spark import get_spark_context

MODELVERSION = "005-ft-cropland-logloss"


class Trainer:
    def __init__(self, settings, modeldir, detector, modelversion):
        self.settings = settings
        self.bands = settings["bands"]
        self.outputlabel = settings["outputlabel"]
        self.modeldir = modeldir
        self.model = None
        self.detector = detector
        self.minsamples = settings.get("minsamples", 3000)
        self.modelversion = modelversion

        # gpu = True if spark else False
        self.gpu = False

        # Create the model directory
        Path(modeldir).mkdir(parents=True, exist_ok=True)

        # Create and save the config
        self.create_config()

        # Input parameters
        trainingfile = settings.get("trainingfile")

        # Log to a file as well
        self.sink = logger.add(
            modeldir / "logfile.log",
            level="DEBUG",
            mode="w",
        )

        logger.info("-" * 50)
        logger.info("Initializing CatBoost trainer ...")
        logger.info("-" * 50)
        logger.info(f"Training file: {trainingfile}")

        # Load a test dataframe to derive some information
        test_df = self._load_df_partial(trainingfile)

        # Get the list of features present in the training data
        self.present_features = Trainer._get_training_df_features(test_df)

    def _train(self, sc, **kwargs):
        """Function to train the base model"""

        # Setup the output directory
        outputdir = self.modeldir

        # Check if all required features are present
        self._check_features()

        # Setup the model
        self.model = self._setup_model()

        # Get and check trainingdata
        logger.info("Preparing training data ...")
        cal_data, val_data, test_data = self._get_trainingdata(
            outputdir,
            settings=self.settings,
            **kwargs,
        )
        self._check_trainingdata(cal_data, val_data, outputdir)

        # Remap to string labels
        label_mapping = self.settings["classes"]
        cal_data["label"] = cal_data["output"].map(label_mapping)
        val_data["label"] = val_data["output"].map(label_mapping)
        test_data["label"] = test_data["output"].map(label_mapping)

        # Save processed data to disk for debugging
        logger.info("Saving processed data ...")
        cal_data.to_parquet(Path(outputdir) / "processed_calibration_df.parquet")

        # Save ref_id counts to config
        self.config["ref_id_counts"] = {}
        self.config["ref_id_counts"]["CAL"] = (
            cal_data["ref_id"].value_counts().to_dict()
        )
        self.config["ref_id_counts"]["VAL"] = (
            val_data["ref_id"].value_counts().to_dict()
        )
        self.config["ref_id_counts"]["TEST"] = (
            test_data["ref_id"].value_counts().to_dict()
        )
        self.save_config()

        # Train the model. If on spark -> run training on executor with multiple cores
        logger.info("Starting training ...")

        def _fit_helper(model, cal_data, val_data):
            logger.info("Start training ...")
            # Setup datapools for training
            calibration_data, eval_data = self._setup_datapools(cal_data, val_data)
            model.fit(
                calibration_data,
                eval_set=eval_data,
                verbose=50,
            )
            logger.info("Finished training ...")

            return model

        # Remove logger to file because otherwise
        # we get serialization issues on spark
        logger.remove(self.sink)

        if sc is None:
            self.model = _fit_helper(self.model, cal_data, val_data)
        else:
            logger.info("Running training on executor ...")
            cal_data_bc = sc.broadcast(cal_data)
            val_data_bc = sc.broadcast(val_data)
            rdd = sc.parallelize([0], numSlices=1)
            self.model = rdd.map(
                lambda _: _fit_helper(self.model, cal_data_bc.value, val_data_bc.value)
            ).collect()[0]

            cal_data_bc.unpersist()
            val_data_bc.unpersist()

        # Add the logger again
        self.sink = logger.add(
            self.modeldir / "logfile.log",
            level="DEBUG",
        )

        # Save the model
        modelname = f"PrestoDownstreamCatBoost_{self.detector}_v{self.modelversion}"
        self.save_model(self.model, outputdir, modelname)

        # Test the model
        self.evaluate(self.model, test_data, outputdir)

        # Plot feature importances
        self._plot_feature_importance(self.model, outputdir)

        logger.success("Base model trained!")

    def train(self, sc=None, **kwargs):
        # train model
        self._train(sc, minsamples=self.minsamples, **kwargs)

    def _load_df(self, file):
        df = pd.read_parquet(Path(file) / f"training_df_{self.outputlabel}.parquet")

        return df

    def _load_df_partial(self, infile, num_rows=100):
        import pyarrow.dataset as ds

        dataset = ds.dataset(infile, format="parquet", partitioning="hive")
        scanner = dataset.to_batches(batch_size=num_rows)

        # Extract rows from the scanner
        rows = []
        for batch in scanner:
            rows.append(batch.to_pandas())
            if (
                len(rows[0]) >= num_rows
            ):  # Stop once we reach the desired number of rows
                break
        df = pd.concat(rows)

        return df

    def evaluate(self, model, testdata, outdir, pattern=""):
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        logger.info("Getting test results ...")

        # In test mode, all valid samples are equal
        idxvalid = testdata["weight"] > 0
        inputs = testdata.loc[idxvalid, self.bands]
        outputs = testdata.loc[idxvalid, "label"]
        orig_outputs = testdata.loc[idxvalid, "orig_output"]

        # Run evaluation
        predictions = model.predict(inputs)

        # Make sure predictions are now 1D
        predictions = predictions.squeeze()

        # Convert labels to the same type
        outputs = outputs.astype(str)
        predictions = predictions.astype(str)

        # Make absolute confusion matrix
        cm = confusion_matrix(outputs, predictions, labels=np.unique(outputs))
        disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(outputs))
        _, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        plt.tight_layout()
        plt.savefig(str(Path(outdir) / f"{pattern}CM_abs.png"))
        plt.close()

        # Make relative confusion matrix
        cm = confusion_matrix(
            outputs, predictions, normalize="true", labels=np.unique(outputs)
        )
        disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(outputs))
        _, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap=plt.cm.Blues, values_format=".1f", colorbar=False)
        plt.tight_layout()
        plt.savefig(str(Path(outdir) / f"{pattern}CM_norm.png"))
        plt.close()

        # Compute evaluation metrics
        metrics = {}
        if len(np.unique(outputs)) == 2:
            metrics["OA"] = np.round(accuracy_score(outputs, predictions), 3)
            metrics["F1"] = np.round(
                f1_score(outputs, predictions, pos_label=self.settings["classes"][1]), 3
            )
            metrics["Precision"] = np.round(
                precision_score(
                    outputs, predictions, pos_label=self.settings["classes"][1]
                ),
                3,
            )
            metrics["Recall"] = np.round(
                recall_score(
                    outputs, predictions, pos_label=self.settings["classes"][1]
                ),
                3,
            )
        else:
            metrics["OA"] = np.round(accuracy_score(outputs, predictions), 3)
            metrics["F1"] = np.round(f1_score(outputs, predictions, average="macro"), 3)
            metrics["Precision"] = np.round(
                precision_score(outputs, predictions, average="macro"), 3
            )
            metrics["Recall"] = np.round(
                recall_score(outputs, predictions, average="macro"), 3
            )

        # Write metrics to disk
        with open(str(Path(outdir) / f"{pattern}metrics.txt"), "w") as f:
            f.write("Test results:\n")
            for key in metrics.keys():
                f.write(f"{key}: {metrics[key]}\n")
                logger.info(f"{key} = {metrics[key]}")

        cm = confusion_matrix(outputs, predictions)
        outputlabels = list(np.unique(outputs))
        predictlabels = list(np.unique(predictions))
        outputlabels.extend(predictlabels)
        outputlabels = list(dict.fromkeys(outputlabels))
        outputlabels.sort()
        cm_df = pd.DataFrame(data=cm, index=outputlabels, columns=outputlabels)
        outfile = Path(outdir) / f"{pattern}confusion_matrix.txt"
        cm_df.to_csv(outfile)

        datadict = {
            "ori": orig_outputs.values,
            "pred": predictions,
        }
        data = pd.DataFrame.from_dict(datadict)
        count = data.groupby(["ori", "pred"]).size()
        result = count.to_frame(name="count").reset_index()
        outfile = Path(outdir) / f"{pattern}confusion_matrix_original_labels.txt"
        result.to_csv(outfile, index=False)

        return metrics

    def save_model(self, model, outputdir, modelname):
        # Both as cbm and onnx
        model.save_model(Path(outputdir) / (modelname + ".cbm"))
        model.save_model(
            f"{Path(outputdir) / (modelname + '.onnx')}",
            format="onnx",
            export_parameters={
                "onnx_domain": "ai.catboost",
                "onnx_model_version": 1,
                "onnx_doc_string": f"Default {self.detector} model using CatBoost",
                "onnx_graph_name": f"CatBoostModel_for_{self.detector}",
            },
        )

    def create_config(self):
        import copy

        config = copy.deepcopy(self.settings)
        config["trainingfile"] = str(config["trainingfile"])
        self.config = config
        self.save_config()

    def save_config(self):
        configpath = Path(self.modeldir) / "config.json"
        with open(configpath, "w") as f:
            json.dump(self.config, f, indent=4)

    @staticmethod
    def _get_training_df_features(df):
        present_features = df.columns.tolist()

        return present_features

    def _setup_model(self):
        # Setup the model
        from catboost import CatBoostClassifier

        logger.info("Setting up model ...")

        # Manually control class name order!
        class_names = [
            self.settings["classes"][class_nr]
            for class_nr in range(len(self.settings["classes"]))
        ]

        model = CatBoostClassifier(
            iterations=8000,
            depth=8,
            class_names=class_names,
            random_seed=1234,
            learning_rate=0.05,
            early_stopping_rounds=50,
            l2_leaf_reg=3,
            eval_metric="Logloss",
            train_dir=self.modeldir,
        )

        # Print a summary of the model
        model_params = model.get_params()
        model_params["train_dir"] = str(model_params["train_dir"])
        self.config["model_params"] = model_params
        self.save_config()
        logger.info(model_params)

        return model

    def _get_trainingdata(self, outputdir, minsamples=500, settings=None, **kwargs):
        settings = self.settings if settings is None else settings

        # Get the data
        cal_data, val_data, test_data = get_training_data(
            self.detector,
            settings,
            self.bands,
            logdir=outputdir,
            minsamples=minsamples,
            **kwargs,
        )

        return cal_data, val_data, test_data

    def _setup_datapools(self, cal_data, val_data):
        # Setup dataset Pool
        calibration_data = Pool(
            data=cal_data[self.bands],
            label=cal_data["label"],
            weight=cal_data["weight"],
        )
        eval_data = Pool(
            data=val_data[self.bands],
            label=val_data["label"],
            weight=val_data["weight"],
        )

        return calibration_data, eval_data

    def _check_trainingdata(self, cal_data, val_data, outputdir):
        # Run some checks
        plt.hist(val_data[self.bands].values.ravel(), 100)
        plt.savefig(Path(outputdir) / ("inputdist_val.png"))
        plt.close()
        plt.hist(cal_data[self.bands].values.ravel(), 100)
        plt.savefig(Path(outputdir) / ("inputdist_cal.png"))
        plt.close()
        logger.info(f"Unique CAL outputs: {np.unique(cal_data['output'])}")
        logger.info(f"Unique VAL outputs: {np.unique(val_data['output'])}")
        logger.info(f"Unique CAL weights: {np.unique(cal_data['weight'])}")
        logger.info(f"Unique VAL weights: {np.unique(val_data['weight'])}")
        logger.info(
            f"Mean Pos. weight: "
            f"{np.mean(cal_data['weight'][cal_data['output'] == 1])}"
        )
        logger.info(
            f"Mean Neg. weight: "
            f"{np.mean(cal_data['weight'][cal_data['output'] == 0])}"
        )
        ratio_pos = np.sum(cal_data["output"] == 1) / cal_data["output"].size
        logger.info(f"Ratio pos/neg outputs: {ratio_pos}")

    def _check_features(self):
        present_features = [ft for ft in self.present_features]

        for band in self.bands:
            if band not in present_features:
                raise RuntimeError(f"Feature `{band}` not found in features.")

    def _plot_feature_importance(self, model, outputdir):
        # Save feature importance plot
        logger.info("Plotting feature importance ...")
        ft_imp = model.get_feature_importance()
        sorting = np.argsort(np.array(ft_imp))[::-1]

        f, ax = plt.subplots(1, 1, figsize=(20, 8))
        ax.bar(np.array(self.bands)[sorting], np.array(ft_imp)[sorting])
        ax.set_xticklabels(np.array(self.bands)[sorting], rotation=90)
        plt.tight_layout()
        plt.savefig(str(Path(outputdir) / "feature_importance.png"))

    @staticmethod
    def write_log_df(metrics, aez, modelname, cal_data, outputdir, parentmetrics=None):
        outfile = Path(outputdir) / "log_df.csv"

        nr_cal_samples = cal_data[0].shape[0]
        logdata = {
            "model": [modelname],
            "aez": [aez],
            "cal_samples": [nr_cal_samples],
            "OA": [metrics["OA"]],
            "OA_parent": [np.nan],
            "F1": [metrics["F1"]],
            "F1_parent": [np.nan],
            "Precision": [metrics["Precision"]],
            "Precision_parent": [np.nan],
            "Recall": [metrics["Recall"]],
            "Recall_parent": [np.nan],
        }

        if parentmetrics is not None:
            logdata["OA_parent"] = [parentmetrics["OA"]]
            logdata["F1_parent"] = [parentmetrics["F1"]]
            logdata["Precision_parent"] = [parentmetrics["Precision"]]
            logdata["Recall_parent"] = [parentmetrics["Recall"]]

        log_df = pd.DataFrame.from_dict(logdata).set_index("model")
        log_df.to_csv(outfile)

    @staticmethod
    def load_log_df(outputdir):
        outfile = Path(outputdir) / "log_df.csv"
        if not outfile.is_file():
            raise FileNotFoundError(f"Logfile `{outfile}` not found.")

        log_df = pd.read_csv(outfile, index_col=0)

        return log_df


def main(detector, trainingsettings, outdir_base, MODELVERSION, sc=None):
    # Plot without display
    plt.switch_backend("Agg")

    logger.info(f'Training on bands: {trainingsettings["bands"]}')

    # Get path to output model directory
    modeldir = Path(outdir_base)

    # Initialize trainer
    trainer = Trainer(trainingsettings, modeldir, detector, MODELVERSION)

    # Train the model;
    trainer.train(sc=sc)

    logger.success("Model trained!")


if __name__ == "__main__":
    spark = False
    localspark = False

    if spark:
        logger.info("Setting up spark ...")
        sc = get_spark_context(localspark=localspark)
    else:
        sc = None

    # Supress debug messages
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Get the trainingsettings
    BANDS_CROPLAND_PRESTO = [f"presto_ft_{i}" for i in range(128)]
    trainingdir = Path(
        "/vitodata/worldcereal/features/preprocessedinputs-monthly-nointerp"
    )

    trainingsettings = {
        "trainingfile": trainingdir
        / "training_df_presto-ss-wc-ft-ct_cropland_CROPLAND2_30D_random_time-token=none_balance=True_augment=True_presto-worldcereal.parquet",
        "outputlabel": "LANDCOVER_LABEL",
        "targetlabels": [11],
        "ignorelabels": [10],
        "focuslabels": [12, 13, 20, 30, 50, 999],
        "focusmultiplier": 3,
        "filter_worldcover": True,
        "classes": {0: "other", 1: "cropland"},
        "bands": BANDS_CROPLAND_PRESTO,
        "pos_neg_ratio": 0.45,
        "minsamples": 500,
    }

    # Output parameters
    detector = "cropland"
    outdir = (
        "/vitodata/worldcereal/models/"
        f"PrestoDownstreamCatBoost/{detector}_detector_"
        f"PrestoDownstreamCatBoost"
        f"_v{MODELVERSION}"
    )

    main(detector, trainingsettings, outdir, MODELVERSION, sc=sc)
