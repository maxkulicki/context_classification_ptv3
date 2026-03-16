"""
Standardized Dataset for Point Cloud Classification

Individual tree point clouds stored per-species, each as a .laz (or .npy) file.
Supports train/val splits defined by text files:
  standardized_dataset_{split}.txt
Each line is a relative path like: "Abies_alba/Frey2022_B1T1.laz"
"""

import os
import numpy as np
import copy
import pointops
import torch
from torch.utils.data import Dataset
from copy import deepcopy

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose


@DATASETS.register_module()
class StandardizedDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/standardized_dataset",
        class_names=None,
        transform=None,
        label_level="species",
        num_points=8192,
        uniform_sampling=True,
        save_record=True,
        test_mode=False,
        test_cfg=None,
        loop=1,
    ):
        super().__init__()
        self.data_root = data_root
        self.class_names = dict(zip(class_names, range(len(class_names))))
        self.label_level = label_level
        self.split = split
        self.num_point = num_points
        self.uniform_sampling = uniform_sampling
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        if test_mode:
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

        record_name = f"standardized_dataset_{self.split}"
        if num_points is not None:
            record_name += f"_{num_points}points"
            if uniform_sampling:
                record_name += "_uniform"
        record_path = os.path.join(self.data_root, f"{record_name}.pth")
        if os.path.isfile(record_path):
            logger.info(f"Loading record: {record_name} ...")
            self.data = torch.load(record_path, weights_only=False)
        else:
            logger.info(f"Preparing record: {record_name} ...")
            self.data = {}
            for idx in range(len(self.data_list)):
                data_name = self.data_list[idx]
                logger.info(f"Parsing data [{idx}/{len(self.data_list)}]: {data_name}")
                self.data[data_name] = self.get_data(idx)
            if save_record:
                torch.save(self.data, record_path)

    def _load_laz_coords(self, path):
        try:
            import laspy
        except ImportError as exc:
            raise ImportError(
                "laspy is required to read .laz files. "
                "Install it or convert .laz to .npy first."
            ) from exc
        las = laspy.read(path)
        # las.x/y/z are already scaled to real coordinates
        coord = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float32)
        return coord

    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        data_name = self.data_list[data_idx]
        if data_name in self.data.keys():
            return copy.deepcopy(self.data[data_name])

        rel_path = data_name.replace("\\", "/")
        class_name = rel_path.split("/")[0]
        if self.label_level == "genus":
            class_name = class_name.split("_", 1)[0]
        if class_name not in self.class_names:
            raise ValueError(f"Unknown class '{class_name}' in split list.")

        data_path = os.path.join(self.data_root, rel_path)
        if data_path.lower().endswith(".npy"):
            data = np.load(data_path).astype(np.float32)
            coord = data[:, 0:3]
            normal = data[:, 3:6] if data.shape[1] >= 6 else None
        else:
            coord = self._load_laz_coords(data_path)
            normal = None

        if self.num_point is not None:
            if self.uniform_sampling:
                with torch.no_grad():
                    mask = pointops.farthest_point_sampling(
                        torch.tensor(coord).float().cuda(),
                        torch.tensor([len(coord)]).long().cuda(),
                        torch.tensor([self.num_point]).long().cuda(),
                    )
                coord = coord[mask.cpu()]
                if normal is not None:
                    normal = normal[mask.cpu()]
            else:
                coord = coord[: self.num_point]
                if normal is not None:
                    normal = normal[: self.num_point]

        category = np.array([self.class_names[class_name]])
        data_dict = dict(coord=coord, category=category)
        if normal is not None:
            data_dict["normal"] = normal
        return data_dict

    def get_data_list(self):
        assert isinstance(self.split, str)
        split_path = os.path.join(
            self.data_root, "standardized_dataset_{}.txt".format(self.split)
        )
        if not os.path.isfile(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")
        with open(split_path, "r") as f:
            data_list = [line.strip() for line in f if line.strip()]
        return data_list

    def get_data_name(self, idx):
        data_idx = idx % len(self.data_list)
        return self.data_list[data_idx]

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        assert idx < len(self.data_list)
        data_dict = self.get_data(idx)
        category = data_dict.pop("category")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))
        for i in range(len(data_dict_list)):
            data_dict_list[i] = self.post_transform(data_dict_list[i])
        data_dict = dict(
            voting_list=data_dict_list,
            category=category,
            name=self.get_data_name(idx),
        )
        return data_dict
