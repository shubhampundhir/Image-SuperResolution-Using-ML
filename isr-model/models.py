import cv2
import matplotlib.pyplot as plt

from isr.model import MODEL_REGISTRY, IsrModel


class InterpolationModel(IsrModel):
    def __init__(self, config):
        super(InterpolationModel, self).__init__(config)

    def train(self, list_of_lr_images, list_of_hr_images):
        print("InterpolationModel needs no training")

    def predict(self, list_of_lr_images):
        predicted_hr_images = []
        for lr_image in list_of_lr_images:
            predicted_hr_images.append(
                cv2.resize(
                    lr_image,
                    (lr_image.shape[1] * 4, lr_image.shape[0] * 4),
                    interpolation=cv2.INTER_CUBIC,
                )
            )
        return predicted_hr_images


# TODO: add more models
