"""
Standardized Dataset for Point Cloud Classification

Individual tree point clouds stored per-species, each as a .laz (or .npy) file.
Supports train/val splits defined by text files:
  standardized_dataset_{split}.txt
Each line is a relative path like: "Abies/Frey2022_B1T1.npy"

Context features (AlphaEarth, Topo, SINR, GeoPlantNet) are optionally loaded
from a features CSV (context_csv parameter). They are injected into each
data dict under source-specific keys:
  ctx_ae    — AlphaEarth (64-dim float32)
  ctx_topo  — Topographic variables (6-dim float32)
  ctx_sinr  — SINR backbone features (256-dim float32)
  ctx_gpn   — GeoPlantNet logit scores (18-dim float32, NaN → 0)

Context features are indexed by filename stem (derived from laz_path in the CSV)
and are never stored in the .pth cache — they are injected at retrieval time.
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

# Context source → column prefix/name in features.csv
_SOURCE_META = {
    "alphaearth": {
        "key": "ctx_ae",
        "col_filter": lambda c: c.startswith("A") and c[1:].isdigit(),
    },
    "topo": {
        "key": "ctx_topo",
        "col_filter": lambda c: c in (
            "elevation", "slope", "northness", "eastness", "tri", "tpi"
        ),
    },
    "sinr": {
        "key": "ctx_sinr",
        "col_filter": lambda c: c.startswith("sinr_"),
    },
    "gpn": {
        "key": "ctx_gpn",
        "col_filter": lambda c: c not in {
            "dataset", "tree_id", "genus", "species",
            "latitude", "longitude", "split", "laz_path",
            "elevation", "slope", "northness", "eastness", "tri", "tpi",
        } and not c.startswith("A") and not c.startswith("sinr_"),
    },
}


def _build_context_lookup(pth_path, sources):
    """Load a pre-built context lookup dict from a .pth file.

    The .pth file is produced by preprocess_context_features.py and contains:
        {filename_stem: {"ctx_ae": np.float32, "ctx_topo": ...,
                         "ctx_sinr": ..., "ctx_gpn": ...}}

    Only the keys corresponding to the requested sources are retained, so
    the returned dict is a subset of the stored data.

    Parameters
    ----------
    pth_path : str
        Path to the pre-built .pth lookup (e.g. snapshot_v1/context_features.pth).
    sources : list[str]
        Which sources to keep; subset of {"alphaearth", "topo", "sinr", "gpn"}.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
    """
    full = torch.load(pth_path, weights_only=False)
    wanted_keys = {_SOURCE_META[src]["key"] for src in sources}
    # Filter to only the requested source keys to save memory
    return {
        stem: {k: v for k, v in entry.items() if k in wanted_keys}
        for stem, entry in full.items()
    }


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
        context_pth=None,
        context_sources=None,
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

        # Context features — loaded separately, never cached in .pth
        if context_pth is not None and context_sources:
            logger.info(
                f"Loading context features from {context_pth} "
                f"(sources: {context_sources})"
            )
            self._context_lookup = _build_context_lookup(context_pth, context_sources)
            logger.info(
                f"Context lookup built: {len(self._context_lookup)} stems indexed."
            )
        else:
            self._context_lookup = None

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

    def _inject_context(self, data_dict, idx):
        """Add context feature tensors to data_dict (if context is configured)."""
        if self._context_lookup is None:
            return data_dict
        rel_path = self.data_list[idx % len(self.data_list)].replace("\\", "/")
        stem = os.path.splitext(os.path.basename(rel_path))[0]
        ctx = self._context_lookup.get(stem)
        if ctx is None:
            raise KeyError(
                f"Context features not found for stem '{stem}'. "
                f"Check that features.csv covers all trees in the split."
            )
        for key, arr in ctx.items():
            # Shape (1, D) so Pointcept's concat collation produces (B, D)
            data_dict[key] = torch.tensor(arr[np.newaxis])  # (1, D)
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
        data_dict = self._inject_context(data_dict, idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        assert idx < len(self.data_list)
        data_dict = self.get_data(idx)
        data_dict = self._inject_context(data_dict, idx)
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
