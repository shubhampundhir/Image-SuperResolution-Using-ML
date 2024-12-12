import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from tqdm import tqdm


from glob import glob
import cv2
from sklearn.model_selection import GroupKFold
from skimage.metrics import mean_squared_error, structural_similarity
from sklearn.metrics import mean_absolute_error


def generate_streaked_gaussian(
    n, direction, std_dev_center=10.0, std_dev_corner=15.0, weight_corner=0.5
):
    x = np.linspace(0, n - 1, n)
    y = np.linspace(0, n - 1, n)
    xv, yv = np.meshgrid(x, y)

    center_x, center_y = (n - 1) / 2, (n - 1) / 2
    gaussian_center = np.exp(
        -(((xv - center_x) ** 2 + (yv - center_y) ** 2) / (2 * std_dev_center**2))
    )

    if direction == 0:  # Top-left
        corner_x, corner_y = 0, 0
    elif direction == 1:  # Top-right
        corner_x, corner_y = n - 1, 0
    elif direction == 2:  # Bottom-left
        corner_x, corner_y = 0, n - 1
    elif direction == 3:  # Bottom-right
        corner_x, corner_y = n - 1, n - 1
    else:
        raise ValueError("Direction must be one of 0, 1, 2, 3")

    gaussian_corner = np.exp(
        -(((xv - corner_x) ** 2 + (yv - corner_y) ** 2) / (2 * std_dev_corner**2))
    )

    streaked_gaussian = (
        1 - weight_corner
    ) * gaussian_center + weight_corner * gaussian_corner

    return streaked_gaussian

class DataManager(object):
    """Class to manage data loading, processing, and cross-validation"""

    def __init__(self, config):
        data_dir = config["data_dir"]
        self.num_folds = config["num_folds"]
        self.config = config
        self.list_of_paths_to_hr_images = sorted(list(glob(data_dir + "/HR/*")))
        self.list_of_paths_to_lr_images = [
            path.replace("/HR/", "/LR/").replace(".png", "x4m.png")
            for path in self.list_of_paths_to_hr_images
        ]
        self.list_of_hr_images = [
            cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            for path in self.list_of_paths_to_hr_images
        ]
        self.list_of_lr_images = [
            cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            for path in self.list_of_paths_to_lr_images
        ]

        self.hr_image_shape = self.list_of_hr_images[0].size
        self.lr_image_shape = self.list_of_lr_images[0].size

        self.df = pd.DataFrame(
            {
                "hr_path": self.list_of_paths_to_hr_images,
                "lr_path": self.list_of_paths_to_lr_images,
                "hr_image": self.list_of_hr_images,
                "lr_image": self.list_of_lr_images,
            }
        )
        self._make_folds()

    def create_patches(self, patch_fraction, stride_fraction):
        """Create image patches"""
        hr_patch_size = int(self.hr_image_shape[0] * patch_fraction)
        lr_patch_size = int(self.lr_image_shape[0] * patch_fraction)
        hr_stride = int(hr_patch_size * stride_fraction)
        lr_stride = int(lr_patch_size * stride_fraction)

        assert (
            hr_patch_size / lr_patch_size == 4
        ), "HR patch size should be 4 times LR patch size"
        assert hr_stride / lr_stride == 4, "HR stride should be 4 times LR stride"

        self.df["hr_patches"] = self.df["hr_image"].apply(
            lambda x: self._create_image_patches(x, patch_size=128, stride=64)
        )
        self.df["lr_patches"] = self.df["lr_image"].apply(
            lambda x: self._create_image_patches(x, patch_size=32, stride=16)
        )
        self.df.loc[:, "patches"] = self.df.apply(
            lambda x: list(zip(x["lr_patches"], x["hr_patches"])), axis=1
        )
        self.df = self.df.explode("patches")
        self.df["lr_patch"] = self.df["patches"].apply(lambda x: x[0])
        self.df["hr_patch"] = self.df["patches"].apply(lambda x: x[1])

        self.df.drop(columns=["lr_patches", "hr_patches", "patches"], inplace=True)

    @staticmethod
    def _create_image_patches(image, patch_size, stride):
        """Create image patches"""
        patches = []
        for i in range(0, image.size[0] - patch_size + 1, stride):
            for j in range(0, image.size[1] - patch_size + 1, stride):
                patch = image.crop((i, j, i + patch_size, j + patch_size))
                patches.append(patch)
        return patches

    def _make_folds(self):
        """Create folds for cross-validation"""
        # NOTE: group k fold because we want all the patches of an image to be in the same fold
        kfold = GroupKFold(n_splits=self.num_folds)
        self.df["fold"] = -1
        for fold, (_, val_idx) in enumerate(
            kfold.split(self.df, groups=self.df["hr_path"])
        ):
            self.df.loc[val_idx, "fold"] = fold

    def get_fold(self, fold):
        """Get data for a specific fold"""
        return self.df[self.df["fold"] != fold], self.df[self.df["fold"] == fold]

class Metrics:
    """Class to manage metrics."""

    @staticmethod
    def mse(true_hr_image, predicted_hr_image):
        return mean_squared_error(true_hr_image, predicted_hr_image)

    @staticmethod
    def ssim(true_hr_image, predicted_hr_image):
        return structural_similarity(
            true_hr_image,
            predicted_hr_image,
            multichannel=True,
            channel_axis=2,
            data_range=1,
        )

    @staticmethod
    def mae(true_hr_image, predicted_hr_image):
        true_hr_image = true_hr_image.reshape(-1)
        predicted_hr_image = predicted_hr_image.reshape(-1)
        return mean_absolute_error(true_hr_image, predicted_hr_image)

    @staticmethod
    def min_max_normalize(image):
        return (image - image.min()) / (image.max() - image.min())

    @staticmethod
    def generate_metric_report(list_of_hr_images, list_of_predictions):
        """Generate metric report."""

        list_of_hr_images = [np.array(hr_image) for hr_image in list_of_hr_images]
        list_of_predictions = [
            np.array(prediction) for prediction in list_of_predictions
        ]

        mae = [
            round(
                Metrics.mae(
                    Metrics.min_max_normalize(hr_image),
                    Metrics.min_max_normalize(prediction),
                ),
                4,
            )
            for hr_image, prediction in zip(list_of_hr_images, list_of_predictions)
        ]
        ssim = [
            round(
                Metrics.ssim(
                    Metrics.min_max_normalize(hr_image),
                    Metrics.min_max_normalize(prediction),
                ),
                4,
            )
            for hr_image, prediction in zip(list_of_hr_images, list_of_predictions)
        ]
        return {"ssim": list(map(float, ssim)), "mae": list(map(float, mae))}


class Patchify:
    def __init__(self, lr_images, hr_images, lr_patch_size):
        self.lr_images = [image / 255.0 for image in lr_images]
        self.hr_images = [image / 255.0 for image in hr_images]
        self.lr_patch_size = lr_patch_size

        # NOTE: dirty padding adds some noise
        dirty_padded_hr_images = []
        for lr_image, hr_image in zip(lr_images, hr_images):
            hr_image = Patchify.dirty_padding(lr_image, hr_image)
            dirty_padded_hr_images.append(hr_image)
        self.hr_images = dirty_padded_hr_images

        # hr patch size is the upscaling factor, automatically inferred from the size of the images
        self.hr_patch_size = hr_images[0].shape[0] // lr_images[0].shape[0]

        self.idx2lr_patches = {}
        self.idx2hr_patches = {}

    @staticmethod
    def dirty_padding(lr_image, hr_image):
        # pad the hr image until its shape is multiple of the lr image dimensions
        lr_image_shape = lr_image.shape
        hr_image_shape = hr_image.shape

        # print("lr image shape", lr_image_shape)
        # print("hr image shape", hr_image_shape)

        padding_x = hr_image_shape[0] % lr_image_shape[0]
        padding_y = hr_image_shape[1] % lr_image_shape[1]

        padding_x = (lr_image_shape[0] - padding_x) if padding_x != 0 else 0
        padding_y = (lr_image_shape[1] - padding_y) if padding_y != 0 else 0
        # print("padding x", padding_x)
        # print("padding y", padding_y)

        if padding_x == 0 and padding_y == 0:
            return hr_image
        hr_image = np.pad(hr_image, ((0, padding_x), (0, padding_y)), mode="constant")
        # print("new hr image shape", hr_image.shape)
        return hr_image

    @staticmethod
    def _pad_image(patch, padding_size, mode):
        # pad the 3 channel image
        return np.pad(
            patch,
            ((padding_size, padding_size), (padding_size, padding_size), (0, 0)),
            mode=mode,
        )

    @staticmethod
    def _extract_lr_patches(image, patch_size, padding_strategy="constant"):
        patches = []
        padding_size = int(patch_size // 2)
        padded_image = Patchify._pad_image(
            image, padding_size=padding_size, mode=padding_strategy
        )
        # get patches such that each pixel is a center of a patch
        for i in range(padding_size, padded_image.shape[0] - padding_size):
            for j in range(padding_size, padded_image.shape[1] - padding_size):
                patch = padded_image[
                    i - padding_size : i + padding_size + 1,
                    j - padding_size : j + padding_size + 1,
                ]
                patches.append(patch)
                # print("extracted patch shape", patch.shape)
        return patches

    def get_lr_patches(self):
        for idx, lr_image in tqdm(enumerate(self.lr_images), total=len(self.lr_images)):
            lr_patches = Patchify._extract_lr_patches(lr_image, self.lr_patch_size)
            self.idx2lr_patches[idx] = lr_patches

        Patchify._check_lr_patching(self.idx2lr_patches, self.lr_images)
        return self.idx2lr_patches

    @staticmethod
    def _extract_hr_patches(image, patch_size):
        patches = []
        for i in range(0, image.shape[0], patch_size):
            for j in range(0, image.shape[1], patch_size):
                patch = image[i : i + patch_size, j : j + patch_size]
                patches.append(patch)

        return patches

    def get_hr_patches(self):
        # the hr images will be of size (lr_image_width * hr_patch_size, lr_image_height * hr_patch_size)
        # make as manu patches as the number of pixels in the lr image each of size (hr_patch_size, hr_patch_size)
        for idx, hr_image in tqdm(enumerate(self.hr_images), total=len(self.hr_images)):
            # print( "num pixels in lr image", self.lr_images[idx].shape[0]*self.lr_images[idx].shape[1])
            # print("num pixels in hr image", hr_image.shape[0]*hr_image.shape[1])
            hr_patches = Patchify._extract_hr_patches(hr_image, self.hr_patch_size)
            # print("num hr patches", len(hr_patches))
            self.idx2hr_patches[idx] = hr_patches

        Patchify._check_hr_patching(
            self.idx2hr_patches, self.hr_images, self.lr_images, self.hr_patch_size
        )
        return self.idx2hr_patches

    @staticmethod
    def _check_hr_patching(idx2hr_patches, hr_images, lr_images, hr_patch_size):
        for idx, hr_image in enumerate(hr_images):
            # print("num hr patches should be equal to num pixels in lr image", len(idx2hr_patches[idx]), lr_images[idx].shape[0]*lr_images[idx].shape[1])
            assert (
                len(idx2hr_patches[idx])
                == lr_images[idx].shape[0] * lr_images[idx].shape[1]
            ), f"{len(idx2hr_patches[idx])} != {lr_images[idx].shape[0]*lr_images[idx].shape[1]}"
            # print("patch size should be equal to upscaling factor", idx2hr_patches[idx][0].shape, (hr_patch_size, hr_patch_size))
            assert idx2hr_patches[idx][0].shape == (
                hr_patch_size,
                hr_patch_size,
                3,
            ), f"{idx2hr_patches[idx][0].shape} != {(hr_patch_size, hr_patch_size)}"

    @staticmethod
    def _check_lr_patching(idx2lr_patches, lr_images):
        for idx, lr_image in enumerate(lr_images):
            assert (
                len(idx2lr_patches[idx]) == lr_image.shape[0] * lr_image.shape[1]
            ), f"{len(idx2lr_patches[idx])} != {lr_image.shape[0]*lr_image.shape[1]}"

    @staticmethod
    def get_image_from_lr_patches(lr_patches, lr_image_shape, patch_size):
        raise NotImplementedError(
            "this function is not needed, you should only use get_image_from_hr_patches"
        )

    @staticmethod
    def get_image_from_hr_patches(hr_patches, lr_image_shape):
        image = []
        for i in range(lr_image_shape[0]):
            image_row = []
            for j in range(lr_image_shape[1]):
                image_row.append(hr_patches[i * lr_image_shape[1] + j])
            image_row = np.concatenate(image_row, axis=1)
            image.append(image_row)
        image = np.concatenate(image, axis=0)
        return image


# ########################## testing the patchify class ############################
# some_train_lr_images, some_train_hr_images = train_lr_images[:5], train_hr_images[:5]
# patchify = Patchify(lr_images=some_train_lr_images, hr_images=some_train_hr_images, lr_patch_size=4)
# idx2lr_patches = patchify.get_lr_patches()
# idx2hr_patches = patchify.get_hr_patches()

# hr_images_reconstructed = []
# for idx, hr_patches in idx2hr_patches.items():
#     hr_image = Patchify.get_image_from_hr_patches(hr_patches, train_lr_images[idx].shape)
#     hr_images_reconstructed.append(hr_image)

# for hr_image, hr_image_reconstructed in zip(train_hr_images, hr_images_reconstructed):
#     plt.subplot(1, 2, 1)
#     plt.imshow(hr_image)
#     plt.subplot(1, 2, 2)
#     plt.imshow(hr_image_reconstructed)
#     plt.title(f"Original HR Image and Reconstructed HR Image, {np.allclose(hr_image, hr_image_reconstructed)}")
#     plt.show()


class SuperResolution(object):
    def __init__(
        self,
        inference_mode=True,
        train_hr_images=None,
        train_lr_images=None,
        val_hr_images=None,
        val_lr_images=None,
        super_resolution_ratio=None,
        train_patch_size=11,
        path_to_models_dir=None,
    ):
        self.super_resolution_ratio = super_resolution_ratio
        self.train_patch_size = train_patch_size
        self.num_models = super_resolution_ratio * super_resolution_ratio * 3

        if not path_to_models_dir:
            self.path_to_models_dir = "models"
            os.makedirs(self.path_to_models_dir, exist_ok=True)
        else:
            self.path_to_models_dir = path_to_models_dir

        self.models = [LinearRegression() for _ in range(self.num_models)]
        self.models2 = [SVR() for _ in range(self.num_models)]

        if inference_mode:
            pass

        else:
            self.train_patchify = Patchify(
                train_lr_images, train_hr_images, train_patch_size
            )
            self.train_idx2lr_patches = self.train_patchify.get_lr_patches()
            self.train_idx2hr_patches = self.train_patchify.get_hr_patches()

            self.val_patchify = Patchify(val_lr_images, val_hr_images, train_patch_size)
            self.val_idx2lr_patches = self.val_patchify.get_lr_patches()
            self.val_idx2hr_patches = self.val_patchify.get_hr_patches()

            self.model_idx2y_train = {}
            self.model_idx2y_val = {}

            for model_idx in range(self.num_models):
                self.model_idx2y_train[model_idx] = []
                self.model_idx2y_val[model_idx] = []
                for hr_patches in tqdm(
                    self.train_idx2hr_patches.values(),
                    total=len(self.train_idx2hr_patches),
                    desc=f"train data for model {model_idx}",
                ):
                    for hr_patch in hr_patches:
                        self.model_idx2y_train[model_idx].append(
                            hr_patch.flatten()[model_idx]
                        )
                self.model_idx2y_train[model_idx] = np.array(
                    self.model_idx2y_train[model_idx]
                )

                for hr_patches in tqdm(
                    self.val_idx2hr_patches.values(),
                    total=len(self.val_idx2hr_patches),
                    desc=f"val data for model {model_idx}",
                ):
                    for hr_patch in hr_patches:
                        self.model_idx2y_val[model_idx].append(
                            hr_patch.flatten()[model_idx]
                        )
                self.model_idx2y_val[model_idx] = np.array(
                    self.model_idx2y_val[model_idx]
                )

            for model_idx, y_train in self.model_idx2y_train.items():
                print(f"model {model_idx} y_train shape", y_train.shape)

            # since the input is the same for all the pixel predictors we can use the same input for all the models
            self.X_train = []
            for lr_patches in tqdm(
                self.train_idx2lr_patches.values(),
                total=len(self.train_idx2lr_patches),
                desc="train features",
            ):
                for lr_patch in lr_patches:
                    self.X_train.append(lr_patch.ravel())
            self.X_val = []
            for lr_patch in tqdm(
                self.val_idx2lr_patches.values(),
                total=len(self.val_idx2lr_patches),
                desc="val features",
            ):
                for lr_patch in lr_patches:
                    self.X_val.append(lr_patch.ravel())
            self.X_train = np.array(self.X_train)
            self.X_val = np.array(self.X_val)

            # scale the data
            self.X_train = self.X_train / 255.0
            self.X_val = self.X_val / 255.0

            print("train model data")
            for model_idx in range(self.num_models):
                for X, y in zip(self.X_train, self.model_idx2y_train[model_idx]):
                    break

            print("val model data")
            for model_idx in range(self.num_models):
                for X, y in zip(self.X_val, self.model_idx2y_val[model_idx]):
                    break

    @staticmethod
    def coerce(patched_image, predicted_hr_image):
        # recale images
        patched_image = cv2.resize(
            patched_image,
            (predicted_hr_image.shape[1], predicted_hr_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        patched_image = (patched_image - np.min(patched_image)) / (
            np.max(patched_image) - np.min(patched_image)
        )
        predicted_hr_image = (predicted_hr_image - np.min(predicted_hr_image)) / (
            np.max(predicted_hr_image) - np.min(predicted_hr_image)
        )
        return patched_image - predicted_hr_image * 0.1

    def dump_models(self):
        for model_idx, model in enumerate(self.models):
            joblib.dump(model, f"{self.path_to_models_dir}/model_{model_idx}.pkl")

    def load_models(self):
        for model_idx in range(self.num_models):
            print(f"loading {self.path_to_models_dir}/model_{model_idx}.pkl")
            self.models[model_idx] = joblib.load(
                f"{self.path_to_models_dir}/model_{model_idx}.pkl"
            )

    def train(self):
        for model_idx in tqdm(
            range(self.num_models), total=self.num_models, desc="training models"
        ):
            self.models[model_idx].fit(self.X_train, self.model_idx2y_train[model_idx])

    def predict(self):
        y_preds = []
        for model_idx in range(self.num_models):
            y_pred = self.models[model_idx].predict(self.X_val)
            y_preds.append(y_pred)
        return y_preds

    def infer_image(self, image):
        lr_patches = Patchify._extract_lr_patches(image, 5)
        X = []
        for lr_patch in lr_patches:
            X.append(lr_patch.ravel())
        print(len(X), X[0].shape)
        y_preds = []
        for model_idx in range(self.num_models):
            y_pred = self.models[model_idx].predict(X)
            y_preds.append(y_pred)
        print(len(y_preds), y_preds[0].shape)
        predicted_image = np.zeros((image.shape[0] * 4, image.shape[1] * 4, 3))
        list_of_pixels = []

        for pixels_48 in zip(*y_preds):
            list_of_pixels.append(pixels_48)

        for i in range(0, image.shape[0] * 4, 4):
            for j in range(0, image.shape[1] * 4, 4):
                for k in range(3):
                    for l in range(4):
                        for m in range(4):
                            predicted_image[i + l, j + m, k] = list_of_pixels[
                                (i // 4) * 25 + (j // 4)
                            ][k * 12 + l * 4 + m]

        # turn all pixels that are white to 0
        for i in range(predicted_image.shape[0]):
            for j in range(predicted_image.shape[1]):
                if np.all(predicted_image[i, j] == np.array([255, 255, 255])):
                    predicted_image[i, j] = np.array([0, 0, 0])

        predicted_image = (predicted_image - np.min(predicted_image)) / (
            np.max(predicted_image) - np.min(predicted_image)
        )

        # rescale to 0, 255
        predicted_image = predicted_image * 255.0
        return SuperResolution.coerce(
            patched_image=image, predicted_hr_image=predicted_image
        )

    def evaluate(self, list_of_images):
        metrics = Metrics()
        list_of_predicted_images = []
        for image in list_of_images:
            hr_image = self.infer_image(image)
            list_of_predicted_images.append(hr_image)
        metric_report = metrics.generate_metric_report(hr_image, image)
        print(metric_report)


# method2ssim = {
#     "Bilinear Interpolation": [],
#     "Nearest Neighbour Interpolation": [],
#     "Bicubic Interpolation": [],
#     "Super Resolution": [],
# }
# method2mae = {
#     "Bilinear Interpolation": [],
#     "Nearest Neighbour Interpolation": [],
#     "Bicubic Interpolation": [],
#     "Super Resolution": [],
# }

# list_of_lr_hr_pairs = [
#     ("data/SuperResolution/LR/0802x4m.png", "data/SuperResolution/HR/0802.png"),
#     ("data/SuperResolution/LR/0803x4m.png", "data/SuperResolution/HR/0803.png"),
#     ("data/SuperResolution/LR/0804x4m.png", "data/SuperResolution/HR/0804.png"),
#     ("data/SuperResolution/LR/0805x4m.png", "data/SuperResolution/HR/0805.png"),
#     ("data/SuperResolution/LR/0806x4m.png", "data/SuperResolution/HR/0806.png"),
# ]

# for path_to_vlidation_lr_image, path_to_vlidation_hr_image in list_of_lr_hr_pairs:
#     lr_image = cv2.imread(path_to_vlidation_lr_image)
#     hr_image = cv2.imread(path_to_vlidation_hr_image)

#     lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
#     hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

#     predicted_hr_image = sisr.infer_image(image=lr_image)
#     metric_report = Metrics.generate_metric_report([hr_image], [predicted_hr_image])
#     method2ssim["Super Resolution"].append(metric_report["ssim"][0])
#     method2mae["Super Resolution"].append(metric_report["mae"][0])

#     # bilinear interpolation
#     predicted_hr_image = cv2.resize(lr_image, (hr_image.shape[1], hr_image.shape[0]), interpolation=cv2.INTER_LINEAR)
#     metric_report = Metrics.generate_metric_report([hr_image], [predicted_hr_image])

#     method2ssim["Bilinear Interpolation"].append(metric_report["ssim"][0])
#     method2mae["Bilinear Interpolation"].append(metric_report["mae"][0])

#     # nearest neighbour interpolation
#     predicted_hr_image = cv2.resize(lr_image, (hr_image.shape[1], hr_image.shape[0]), interpolation=cv2.INTER_NEAREST)
#     metric_report = Metrics.generate_metric_report([hr_image], [predicted_hr_image])

#     method2ssim["Nearest Neighbour Interpolation"].append(metric_report["ssim"][0])
#     method2mae["Nearest Neighbour Interpolation"].append(metric_report["mae"][0])

#     # bicubic interpolation
#     predicted_hr_image = cv2.resize(lr_image, (hr_image.shape[1], hr_image.shape[0]), interpolation=cv2.INTER_CUBIC)
#     metric_report = Metrics.generate_metric_report([hr_image], [predicted_hr_image])

#     method2ssim["Bicubic Interpolation"].append(metric_report["ssim"][0])
#     method2mae["Bicubic Interpolation"].append(metric_report["mae"][0])


# df_ssim = pd.DataFrame(method2ssim)
# df_mae = pd.DataFrame(method2mae)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process paths to low-resolution and high-resolution images."
    )
    parser.add_argument(
        "--low_resolution_image",
        type=str,
        required=True,
        help="Path to the low-resolution image.",
    )
    parser.add_argument(
        "--high_resolution_image",
        type=str,
        required=True,
        help="Path to the high-resolution image.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    print(f"Low-resolution image path: {args.low_resolution_image}")
    print(f"High-resolution image path: {args.high_resolution_image}")

    sisr = SuperResolution(inference_mode=True, super_resolution_ratio=4)
    sisr.load_models()

    lr_image = cv2.imread(args.low_resolution_image)
    hr_image = cv2.imread(args.high_resolution_image)

    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
    hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
    predicted_hr_image = sisr.infer_image(image=lr_image)
    cv2.imwrite("predicted_hr_image.png", predicted_hr_image)

    # generat metrics
    metric_report = Metrics.generate_metric_report(
        list_of_hr_images=[hr_image], list_of_predictions=[predicted_hr_image]
    )

    plt.rcParams["figure.figsize"] = (15, 5)
    plt.rcParams["figure.dpi"] = 400
    plt.subplot(1, 3, 1)
    plt.imshow(lr_image)
    plt.title("LR Image")
    plt.subplot(1, 3, 2)
    plt.imshow(predicted_hr_image)
    plt.title("Predicted HR Image")
    plt.subplot(1, 3, 3)
    plt.imshow(hr_image)
    plt.title("True HR Image")
    plt.suptitle(f"Predicted HR Image and True HR Image {metric_report}")
    plt.tight_layout()
    plt.savefig("predicted_hr_image.png")
    plt.show()
