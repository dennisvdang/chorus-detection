#!/usr/bin/env python
"""Freeze a reproducible train/val/test song split to data/splits/.

Song IDs are sorted (not filesystem order) then shuffled with a fixed seed, so
every trial and both grid preprocessings share identical membership. This
replaces the os.listdir-order-dependent split in
pytorch_core/data/dataset.create_data_loaders for the ablation study.
"""

import argparse
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(REPO_ROOT, "data", "clean_labeled.csv"))
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data", "splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    args = parser.parse_args()

    ids = sorted(str(s) for s in pd.read_csv(args.csv)["SongID"].unique())
    rng = np.random.RandomState(args.seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    splits = {
        "train_songs.txt": ids[:n_train],
        "val_songs.txt": ids[n_train:n_train + n_val],
        "test_songs.txt": ids[n_train + n_val:],
    }
    os.makedirs(args.out_dir, exist_ok=True)
    for name, group in splits.items():
        with open(os.path.join(args.out_dir, name), "w") as f:
            f.write("\n".join(group) + "\n")
        print(f"{name}: {len(group)} songs")


if __name__ == "__main__":
    main()
