from isr.experiment import Experiment


def main():
    config = {
        "data": {
            "data_dir": "data/SuperResolution",
            "num_folds": 5,
        },
        "model": {
            "model_type": "InterpolationModel",
        },
        "path_to_artifacts_dir": "artifacts",
    }
    experiment_obj = Experiment(config)
    experiment_obj.run()


if __name__ == "__main__":
    main()
