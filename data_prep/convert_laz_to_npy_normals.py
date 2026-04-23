"""
Convert snapshot_1 LAZ files to NPY with normals.

Pipeline per tree:
  CPU workers (parallel): load LAZ -> voxel downsample -> estimate normals
  Main process (GPU):     FPS to n_final, carrying normals along -> save (N, 6)

Output arrays: float32 (N, 6) = [x, y, z, nx, ny, nz]
"""

import argparse
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import laspy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--voxel_size", type=float, default=0.05)
    p.add_argument("--n_final", type=int, default=8192)
    p.add_argument("--normal_k", type=int, default=20)
    p.add_argument("--workers", type=int, default=60)
    p.add_argument("--split_root", default=None)
    return p.parse_args()


# ── CPU worker ────────────────────────────────────────────────────────────────

def _voxel_downsample(coords: np.ndarray, voxel_size: float) -> np.ndarray:
    grid = (coords / voxel_size).astype(np.int64)
    _, idx = np.unique(grid, axis=0, return_index=True)
    return coords[idx]


def _compute_normals(coords: np.ndarray, k: int) -> np.ndarray:
    from scipy.spatial import KDTree
    k = min(k, len(coords))
    tree = KDTree(coords)
    _, idxs = tree.query(coords, k=k)                        # (N, k)
    neighbors = coords[idxs]                                  # (N, k, 3)
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.einsum('nkd,nke->nde', centered, centered)      # (N, 3, 3)
    _, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, :, 0].astype(np.float32)               # smallest eigenvec


def process_file_cpu(args_tuple):
    """Load, voxel-downsample, estimate normals. Returns arrays for GPU FPS."""
    in_path, out_path, voxel_size, normal_k, overwrite = args_tuple

    if Path(out_path).exists() and not overwrite:
        return out_path, None, None, None  # skipped

    t0 = time.time()
    las = laspy.read(in_path)
    coord = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float32)
    coord = _voxel_downsample(coord, voxel_size)
    normals = _compute_normals(coord, k=normal_k)
    return out_path, coord, normals, time.time() - t0


# ── GPU FPS (main process only) ───────────────────────────────────────────────

def _fps_gpu(coords: np.ndarray, n: int):
    """Returns index array of length n (or len(coords) if already small)."""
    import torch
    import pointops
    N = coords.shape[0]
    if N <= n:
        return np.arange(N)
    pts = torch.from_numpy(coords).float().cuda(non_blocking=True)
    idx = pointops.farthest_point_sampling(
        pts,
        torch.tensor([N], device=pts.device, dtype=torch.long),
        torch.tensor([n], device=pts.device, dtype=torch.long),
    )
    return idx.cpu().numpy()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for class_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        for in_path in sorted(class_dir.glob("*.laz")):
            out_path = output_root / class_dir.name / (in_path.stem + ".npy")
            tasks.append((in_path, out_path,
                          args.voxel_size, args.normal_k, args.overwrite))

    total = len(tasks)
    print(f"Found {total} files | workers={args.workers} | "
          f"voxel={args.voxel_size}m | k={args.normal_k} | n_final={args.n_final}")

    t_start = time.time()
    done = 0

    # Pool is created BEFORE importing torch/CUDA so workers don't inherit
    # a CUDA context (fork-safe).
    with Pool(processes=args.workers) as pool:
        # Lazily import GPU libs after fork
        import torch   # noqa: F401  (triggers CUDA init in main process only)
        import pointops  # noqa: F401

        for out_path, coord, normals, cpu_time in \
                pool.imap_unordered(process_file_cpu, tasks):
            done += 1
            out_path = Path(out_path)

            if coord is None:
                if done % 500 == 0:
                    print(f"[{done}/{total}] (skipped existing)")
                continue

            # GPU FPS on xyz, then apply same indices to normals
            t_gpu = time.time()
            idx = _fps_gpu(coord, args.n_final)
            gpu_time = time.time() - t_gpu

            out = np.concatenate([coord[idx], normals[idx]], axis=1)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, out)

            elapsed_total = time.time() - t_start
            rate = done / elapsed_total
            eta_min = (total - done) / rate / 60
            print(f"[{done}/{total}] {out_path.parent.name}/{out_path.name}"
                  f"  cpu={cpu_time:.1f}s  gpu={gpu_time:.2f}s"
                  f"  ETA={eta_min:.1f}min")

    # Copy split txt files
    split_root = Path(args.split_root) if args.split_root else input_root
    for split_name in ["standardized_dataset_train.txt", "standardized_dataset_val.txt"]:
        src = split_root / split_name
        if src.exists():
            (output_root / split_name).write_text(src.read_text())
            print(f"Copied {split_name}")

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Output: {output_root}")


if __name__ == "__main__":
    main()
