import pickle
from abc import ABC

import numpy as np
from PIL import Image

"""🚨 DO NOT MODIFY THIS FILE"""

MODEL_REGISTRY = {}


class RegistryMeta(type):
    def __init__(cls, name, bases, attrs):
        super(RegistryMeta, cls).__init__(name, bases, attrs)
        if name != "IsrModel":
            MODEL_REGISTRY[name] = cls


class IsrModel(metaclass=RegistryMeta):
    """Interface for different ISR models."""

    def __init__(self, config):
        self.model = None
        self.config = config

    def load(self, filename):
        """Load model from file"""
        with open(filename, "rb") as f:
            self.model = pickle.load(f)

    def save(self, filename):
        """Save model to file"""
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)

    def train(self, list_of_lr_images, list_of_hr_images):
        """Train model"""
        raise NotImplementedError

    def predict(self, list_of_lr_images):
        """Predict"""
        raise NotImplementedError

    def convert_prediction_to_image(self, prediction):
        """Convert prediction to image"""
        if isinstance(prediction, np.ndarray):
            return Image.fromarray(prediction)
        return prediction
