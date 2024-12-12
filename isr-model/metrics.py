import numpy as np
from skimage.metrics import mean_squared_error, structural_similarity
from sklearn.metrics import mean_absolute_error


class Metrics:
    """Class to manage metrics."""

    @staticmethod
    def mse(true_hr_image, predicted_hr_image):
        return mean_squared_error(true_hr_image, predicted_hr_image)

    @staticmethod
    def ssim(true_hr_image, predicted_hr_image):
        return structural_similarity(
            true_hr_image, predicted_hr_image, multichannel=True, channel_axis=2
        )

    @staticmethod
    def mae(true_hr_image, predicted_hr_image):
        true_hr_image = true_hr_image.reshape(-1)
        predicted_hr_image = predicted_hr_image.reshape(-1)
        return mean_absolute_error(true_hr_image, predicted_hr_image)

    @staticmethod
    def generate_metric_report(list_of_hr_images, list_of_predictions):
        """Generate metric report."""

        # typecast according to the metrics input is list of PIL images
        list_of_hr_images = [np.array(hr_image) for hr_image in list_of_hr_images]
        list_of_predictions = [
            np.array(prediction) for prediction in list_of_predictions
        ]

        mae = [
            Metrics.mae(hr_image, prediction)
            for hr_image, prediction in zip(list_of_hr_images, list_of_predictions)
        ]
        ssim = [
            Metrics.ssim(hr_image, prediction)
            for hr_image, prediction in zip(list_of_hr_images, list_of_predictions)
        ]
        return {"ssim": list(map(float, ssim)), "mae": list(map(float, mae))}
