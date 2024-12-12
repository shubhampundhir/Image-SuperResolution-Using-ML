import json
import logging
import os
from datetime import datetime

import numpy as np

from isr.data import DataManager
from isr.metrics import Metrics
from isr.models import MODEL_REGISTRY

NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class Experiment(object):
    def __init__(self, config):
        self.data_config = config["data"]
        self.model_config = config["model"]
        self.data_manager = DataManager(self.data_config)
        self.model = MODEL_REGISTRY[self.model_config["model_type"]](config)
        path_to_artifacts_dir = config.get("path_to_artifacts_dir", "artifacts")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())

        if not os.path.exists(path_to_artifacts_dir):
            os.mkdir(path_to_artifacts_dir)
        os.mkdir(f"{path_to_artifacts_dir}/{NOW}")
        os.mkdir(f"{path_to_artifacts_dir}/{NOW}/predictions")
        os.mkdir(f"{path_to_artifacts_dir}/{NOW}/trained_model")
        self.path_to_artifacts_dir = path_to_artifacts_dir

        self.logger.addHandler(
            logging.FileHandler(f"{path_to_artifacts_dir}/{NOW}/experiment.log")
        )
        self.model_report = {
            "start_time": None,
            "config": config,
            "end_time": None,
            "metric_reports": {},
            "path_to_trained_model": f"{path_to_artifacts_dir}/{NOW}/trained_model",
            "path_to_predictions": f"{path_to_artifacts_dir}/{NOW}/predictions",
        }
        self.logger.info(f"Data and model initialized using config: {config}")

    def run(self):
        self.model_report["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for fold in range(self.data_config["num_folds"]):
            self.logger.info(f"Running fold {fold}")

            train_df, val_df = self.data_manager.get_fold(fold)
            train_lr_images = train_df["lr_image"]
            train_hr_images = train_df["hr_image"]
            val_lr_images = val_df["lr_image"]
            val_hr_images = val_df["hr_image"]
            self.logger.info(f"\tData loaded for fold {fold}")

            self.model.train(train_lr_images, train_hr_images)
            self.logger.info(f"\tModel trained on fold {fold}")

            predictions = self.model.predict(val_lr_images)
            self.logger.info(f"\tPredictions made on fold {fold}")

            metric_report = Metrics.generate_metric_report(val_hr_images, predictions)

            self.logger.info(
                f"\tMetric report generated for fold {fold}: {metric_report}"
            )
            self.model_report["metric_reports"][fold] = {
                "mae": np.mean(metric_report["mae"]),
                "ssim": np.mean(metric_report["ssim"]),
            }

            path_to_save_predictions_this_fold = os.path.join(
                f"{self.path_to_artifacts_dir}/{NOW}/predictions/fold_{fold}"
            )
            os.mkdir(path_to_save_predictions_this_fold)
            for i, (prediction, val_hr_image) in enumerate(
                zip(predictions, val_hr_images)
            ):
                prediction = self.model.convert_prediction_to_image(prediction)
                val_hr_image = self.model.convert_prediction_to_image(val_hr_image)
                prediction.save(
                    os.path.join(f"{path_to_save_predictions_this_fold}/{i}_pred.png")
                )
                val_hr_image.save(
                    os.path.join(f"{path_to_save_predictions_this_fold}/{i}_true.png")
                )

            self.logger.info(
                f"\tPredictions saved for fold {fold} at {path_to_save_predictions_this_fold}"
            )

            model_name = self.model_config["model_type"]
            path_to_save_trained_model_this_fold = os.path.join(
                f"{self.path_to_artifacts_dir}/{NOW}/trained_model/{model_name}_fold_{fold}.pkl"
            )
            self.model.save(path_to_save_trained_model_this_fold)
            self.logger.info(
                f"\tModel saved for fold {fold} at {path_to_save_trained_model_this_fold}"
            )

        self.model_report["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(
            os.path.join(f"{self.path_to_artifacts_dir}/{NOW}/model_report.json"), "w"
        ) as f:
            json.dump(self.model_report, f)
        self.logger.info(
            f"Model report saved at {self.path_to_artifacts_dir}/{NOW}/model_report.json"
        )
        self.logger.info(
            f"Experiment artifacts saved at {self.path_to_artifacts_dir}/{NOW}"
        )
        time_taken = datetime.strptime(
            self.model_report["end_time"], "%Y-%m-%d %H:%M:%S"
        ) - datetime.strptime(self.model_report["start_time"], "%Y-%m-%d %H:%M:%S")
        self.logger.info(
            f"Experiment completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Time taken: {time_taken}"
        )
