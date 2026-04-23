"""Generate confusion matrix from the trained model on the test set."""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load test data from the cached .pth record
data_dir = "/workspace/Pointcept/data/treescanpl"
exp_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/Pointcept/exp/treescanpl/cls-ptv3-v1m1-0-base"
print(f"Experiment dir: {exp_dir}")

class_names = [
    "Abies", "Acer", "Alnus", "Betula", "Carpinus",
    "Fagus", "Larix", "Picea", "Pinus", "Quercus", "Tilia",
]
class_to_idx = {name: i for i, name in enumerate(class_names)}

# Load test split
split_path = os.path.join(data_dir, "treescanpl_test.txt")
test_samples = np.loadtxt(split_path, dtype="str")

# Load cached test data
# Try to find the cached record
cache_candidates = [f for f in os.listdir(data_dir) if f.startswith("treescanpl_test") and f.endswith(".pth")]
if cache_candidates:
    cache_path = os.path.join(data_dir, cache_candidates[0])
    print(f"Loading cached data: {cache_path}")
    cached_data = torch.load(cache_path, weights_only=False)
else:
    print("No cache found, loading from txt files...")
    cached_data = {}
    for name in test_samples:
        data_shape = "_".join(name.split("_")[0:-1])
        path = os.path.join(data_dir, data_shape, name + ".txt")
        data = np.loadtxt(path, delimiter=",").astype(np.float32)
        coord, normal = data[:, 0:3], data[:, 3:6]
        category = np.array([class_to_idx[data_shape]])
        cached_data[name] = dict(coord=coord, normal=normal, category=category)

# Load model
sys.path.insert(0, "/workspace/Pointcept")
from pointcept.models import build_model
from pointcept.datasets.transform import Compose
from pointcept.utils.config import Config

cfg = Config.fromfile(os.path.join(exp_dir, "config.py"))
model = build_model(cfg.model)
checkpoint = torch.load(os.path.join(exp_dir, "model", "model_best.pth"), weights_only=False)
model.load_state_dict(checkpoint["state_dict"])
model.cuda().eval()

# Define transform (same as val)
transform = Compose([
    dict(type="NormalizeCoord"),
    dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "category"), feat_keys=["coord", "normal"]),
])

all_preds = []
all_labels = []

for i, name in enumerate(test_samples):
    data_shape = "_".join(name.split("_")[0:-1])

    if name in cached_data:
        data_dict = cached_data[name]
        if isinstance(data_dict["category"], np.ndarray):
            label = int(data_dict["category"][0])
        else:
            label = int(data_dict["category"])
    else:
        continue

    import copy
    data_dict_copy = copy.deepcopy(data_dict)
    data_dict_copy = transform(data_dict_copy)

    # Add batch dimension
    for key in data_dict_copy:
        if isinstance(data_dict_copy[key], torch.Tensor):
            data_dict_copy[key] = data_dict_copy[key].unsqueeze(0).cuda()

    # Need offset for the model
    data_dict_copy["offset"] = torch.tensor([data_dict_copy["coord"].shape[1]], dtype=torch.long).cuda()
    data_dict_copy["coord"] = data_dict_copy["coord"].squeeze(0)
    data_dict_copy["grid_coord"] = data_dict_copy["grid_coord"].squeeze(0)
    data_dict_copy["feat"] = data_dict_copy["feat"].squeeze(0)
    data_dict_copy["category"] = torch.tensor([label], dtype=torch.long).cuda()

    with torch.no_grad():
        output = model(data_dict_copy)

    pred = output["cls_logits"].argmax(dim=1).cpu().item()
    all_preds.append(pred)
    all_labels.append(label)

    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(test_samples)}")

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

# Plot - raw counts
fig, axes = plt.subplots(1, 2, figsize=(22, 9))

disp1 = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp1.plot(ax=axes[0], cmap="Blues", values_format="d", xticks_rotation=45)
axes[0].set_title("Confusion Matrix (counts)")

# Plot - normalized by true label (recall per class)
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
disp2.plot(ax=axes[1], cmap="Blues", values_format=".2f", xticks_rotation=45)
axes[1].set_title("Confusion Matrix (normalized by true class)")

plt.tight_layout()
out_path = os.path.join(exp_dir, "confusion_matrix.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to {out_path}")

# Also print text version
print("\nConfusion matrix (rows=true, cols=predicted):")
header = f"{'':>12}" + "".join(f"{n:>10}" for n in class_names)
print(header)
for i, name in enumerate(class_names):
    row = f"{name:>12}" + "".join(f"{cm[i,j]:>10}" for j in range(len(class_names)))
    print(row)

print(f"\nOverall accuracy: {(all_preds == all_labels).mean():.4f}")
print(f"Per-class accuracy (mAcc): {np.mean([cm[i,i]/(cm[i].sum()+1e-10) for i in range(len(class_names))]):.4f}")
