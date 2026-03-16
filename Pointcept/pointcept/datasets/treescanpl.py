"""
TreeScanPL Dataset for Point Cloud Classification

Individual tree point clouds extracted from TreeScanPL forest plots,
with species labels. Format: .npy binary files with x,y,z,nx,ny,nz columns.

Author: Adapted from ModelNetDataset by Xiaoyang Wu
"""

import csv
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
class TreeScanPLDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/treescanpl",
        class_names=None,
        transform=None,
        num_points=8192,
        uniform_sampling=True,
        save_record=True,
        test_mode=False,
        test_cfg=None,
        loop=1,
        fold=None,
    ):
        super().__init__()
        self.data_root = data_root
        self.class_names = dict(zip(class_names, range(len(class_names))))
        self.split = split
        self.fold = fold
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

        # check, prepare record
        if self.fold is not None:
            record_name = f"treescanpl_fold{self.fold}_{self.split}"
        else:
            record_name = f"treescanpl_{self.split}"
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

    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        data_name = self.data_list[data_idx]
        if data_name in self.data.keys():
            return copy.deepcopy(self.data[data_name])
        else:
            data_shape = "_".join(data_name.split("_")[0:-1])
            data_path = os.path.join(
                self.data_root, data_shape, self.data_list[data_idx] + ".npy"
            )
            data = np.load(data_path).astype(np.float32)
            if self.num_point is not None:
                if self.uniform_sampling:
                    with torch.no_grad():
                        mask = pointops.farthest_point_sampling(
                            torch.tensor(data).float().cuda(),
                            torch.tensor([len(data)]).long().cuda(),
                            torch.tensor([self.num_point]).long().cuda(),
                        )
                    data = data[mask.cpu()]
                else:
                    data = data[: self.num_point]
            coord, normal = data[:, 0:3], data[:, 3:6]
            category = np.array([self.class_names[data_shape]])
            return dict(coord=coord, normal=normal, category=category)

    def get_data_list(self):
        assert isinstance(self.split, str)
        if self.fold is not None:
            split_path = os.path.join(
                self.data_root, f"treescanpl_fold{self.fold}_{self.split}.txt"
            )
        else:
            split_path = os.path.join(
                self.data_root, "treescanpl_{}.txt".format(self.split)
            )
        data_list = np.loadtxt(split_path, dtype="str")
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


@DATASETS.register_module()
class TreeScanPLContextDataset(TreeScanPLDataset):
    """TreeScanPL dataset with per-plot AlphaEarth context embeddings.

    Loads AlphaEarth satellite embeddings (64-dim) per plot, maps each sample
    to its plot via sample_plotid_mapping.csv, and injects a 'context' key
    into the data dict. Context is stored as shape (1, context_dim) so that
    collate_fn (torch.cat) produces (B, context_dim).
    """

    def __init__(
        self,
        context_embeddings_path=None,
        sample_plotid_path=None,
        **kwargs,
    ):
        self.context_embeddings_path = context_embeddings_path
        self.sample_plotid_path = sample_plotid_path
        self.context_lookup = None  # populated in _load_context after super().__init__
        super().__init__(**kwargs)
        self.context_lookup = self._load_context(
            context_embeddings_path, sample_plotid_path
        )

    def _load_context(self, embeddings_path, mapping_path):
        """Build sample_name -> normalized 64-dim context vector lookup.

        1. Load sample_plotid_mapping.csv -> dict{sample_name: plot_id}
        2. Load plots_alphaearth_2018.csv -> dict{plot_id: np.array(64,)}
        3. Compose: sample_name -> 64-dim float32 vector
        4. Normalize to zero mean, unit variance (global stats across all plots)
        """
        logger = get_root_logger()

        if embeddings_path is None or mapping_path is None:
            logger.warning("Context paths not provided, context will be zeros.")
            return None

        # Resolve paths relative to data_root if not absolute
        if not os.path.isabs(embeddings_path):
            embeddings_path = os.path.join(self.data_root, embeddings_path)
        if not os.path.isabs(mapping_path):
            mapping_path = os.path.join(self.data_root, mapping_path)

        # Load sample -> plot_id mapping
        sample_to_plot = {}
        with open(mapping_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_to_plot[row["sample_name"]] = int(row["plot_id"])

        # Load plot -> embedding (semicolon-separated CSV)
        ae_cols = [f"A{i:02d}" for i in range(64)]
        plot_embeddings = {}
        with open(embeddings_path, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                plot_id = int(row["num"])
                embedding = np.array(
                    [float(row[c]) for c in ae_cols], dtype=np.float32
                )
                plot_embeddings[plot_id] = embedding

        # Compute global normalization stats across all plot embeddings
        all_embeddings = np.stack(list(plot_embeddings.values()))  # (N_plots, 64)
        emb_mean = all_embeddings.mean(axis=0)
        emb_std = all_embeddings.std(axis=0)
        emb_std[emb_std < 1e-8] = 1.0  # avoid division by zero

        # Compose: sample_name -> normalized (1, 64) vector
        context_lookup = {}
        missing = 0
        for sample_name in self.data_list:
            if sample_name not in sample_to_plot:
                missing += 1
                # Fallback: zero vector for unmapped samples
                context_lookup[sample_name] = np.zeros(
                    (1, len(ae_cols)), dtype=np.float32
                )
                continue
            plot_id = sample_to_plot[sample_name]
            if plot_id not in plot_embeddings:
                missing += 1
                context_lookup[sample_name] = np.zeros(
                    (1, len(ae_cols)), dtype=np.float32
                )
                continue
            emb = (plot_embeddings[plot_id] - emb_mean) / emb_std
            context_lookup[sample_name] = emb.reshape(1, -1)  # (1, 64)

        if missing > 0:
            logger.warning(
                f"Context: {missing}/{len(self.data_list)} samples missing "
                f"embeddings (using zeros)."
            )
        logger.info(
            f"Loaded AlphaEarth context for {len(self.data_list) - missing}/"
            f"{len(self.data_list)} samples from {len(plot_embeddings)} plots."
        )
        return context_lookup

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        if self.context_lookup is not None:
            data_name = self.data_list[idx % len(self.data_list)]
            data_dict["context"] = self.context_lookup[data_name].copy()
        return data_dict


@DATASETS.register_module()
class TreeScanPLFullContextDataset(TreeScanPLContextDataset):
    """TreeScanPL dataset with AlphaEarth context + BDL fertility/moisture.

    Extends TreeScanPLContextDataset by adding one-hot encoded fertility
    (4 classes) and moisture (2 classes) as a 6-dim 'bdl_features' key.
    """

    FERTILITY_CLASSES = ["eutrophic", "mesoeutrophic", "mesotrophic", "oligotrophic"]
    MOISTURE_CLASSES = ["fresh", "moist_or_wet"]

    def __init__(self, bdl_csv_path=None, **kwargs):
        self.bdl_csv_path = bdl_csv_path
        self.bdl_lookup = None
        super().__init__(**kwargs)
        self.bdl_lookup = self._load_bdl(bdl_csv_path)

    def _load_bdl(self, bdl_csv_path):
        """Build sample_name -> one-hot [fertility(4) + moisture(2)] lookup."""
        logger = get_root_logger()

        if bdl_csv_path is None:
            logger.warning("BDL CSV path not provided, bdl_features will be zeros.")
            return None

        if not os.path.isabs(bdl_csv_path):
            bdl_csv_path = os.path.join(self.data_root, bdl_csv_path)

        # Load plot -> (fertility, moisture) mapping
        plot_bdl = {}
        with open(bdl_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                plot_id = int(row["plot_id"])
                plot_bdl[plot_id] = (row["fertility"], row["moisture"])

        # Load sample -> plot_id mapping (already loaded by parent)
        sample_to_plot = {}
        mapping_path = self.sample_plotid_path
        if not os.path.isabs(mapping_path):
            mapping_path = os.path.join(self.data_root, mapping_path)
        with open(mapping_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_to_plot[row["sample_name"]] = int(row["plot_id"])

        fert_to_idx = {c: i for i, c in enumerate(self.FERTILITY_CLASSES)}
        moist_to_idx = {c: i for i, c in enumerate(self.MOISTURE_CLASSES)}
        bdl_dim = len(self.FERTILITY_CLASSES) + len(self.MOISTURE_CLASSES)

        bdl_lookup = {}
        missing = 0
        for sample_name in self.data_list:
            plot_id = sample_to_plot.get(sample_name)
            if plot_id is None or plot_id not in plot_bdl:
                missing += 1
                bdl_lookup[sample_name] = np.zeros((1, bdl_dim), dtype=np.float32)
                continue
            fert, moist = plot_bdl[plot_id]
            vec = np.zeros(bdl_dim, dtype=np.float32)
            if fert in fert_to_idx:
                vec[fert_to_idx[fert]] = 1.0
            if moist in moist_to_idx:
                vec[len(self.FERTILITY_CLASSES) + moist_to_idx[moist]] = 1.0
            bdl_lookup[sample_name] = vec.reshape(1, -1)

        if missing > 0:
            logger.warning(
                f"BDL: {missing}/{len(self.data_list)} samples missing "
                f"fertility/moisture (using zeros)."
            )
        logger.info(
            f"Loaded BDL fertility/moisture for {len(self.data_list) - missing}/"
            f"{len(self.data_list)} samples."
        )
        return bdl_lookup

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        if self.bdl_lookup is not None:
            data_name = self.data_list[idx % len(self.data_list)]
            data_dict["bdl_features"] = self.bdl_lookup[data_name].copy()
        return data_dict