from glob import glob

import cv2
import pandas as pd
from sklearn.model_selection import GroupKFold


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
