import os

import numpy as np

import pandas as pd

import torch

from torch.utils.data import Dataset





class FluidPhysicsDatasetLatent(Dataset):

    """5-channel flow dataset with reusable condition normalization stats."""

    def __init__(self, feature_files, npy_dir, stats_path,

                 cond_stats_path=None, save_cond_stats=False):

        self.npy_dir = npy_dir

        stats = np.load(stats_path)

        self.mean_img = stats['mean'].astype(np.float32)

        self.std_img = stats['std'].astype(np.float32)



        self.feature_names = list(feature_files.keys())

        self.meta = None

        for feat_name, file_path in feature_files.items():

            df = pd.read_csv(

                file_path, sep=r'\s+', header=None,

                names=["sim_id", feat_name, "time"]

            )

            if self.meta is None:

                self.meta = df.copy()

            else:

                self.meta = pd.merge(self.meta, df[["sim_id", feat_name]], on="sim_id")



        self.meta = self.meta.sort_values(by="sim_id").reset_index(drop=True)

        self.cond_columns = self.feature_names + ["time"]

        cond_features = self.meta[self.cond_columns].values.astype(np.float32)



        if cond_stats_path and os.path.exists(cond_stats_path) and not save_cond_stats:

            cond_stats = np.load(cond_stats_path)

            self.cond_mean = cond_stats['mean'].astype(np.float32)

            self.cond_std = cond_stats['std'].astype(np.float32)

            saved_names = cond_stats.get('columns', None)

            if saved_names is not None:

                saved_names = [str(x) for x in saved_names.tolist()]

                if saved_names != self.cond_columns:

                    raise ValueError(

                        f"Condition column mismatch. saved={saved_names}, current={self.cond_columns}"

                    )

        else:

            self.cond_mean = np.mean(cond_features, axis=0).astype(np.float32)

            self.cond_std = np.std(cond_features, axis=0).astype(np.float32)

            self.cond_std = np.where(self.cond_std == 0, 1e-8, self.cond_std).astype(np.float32)

            if cond_stats_path and save_cond_stats:

                os.makedirs(os.path.dirname(cond_stats_path) or '.', exist_ok=True)

                np.savez(

                    cond_stats_path,

                    mean=self.cond_mean,

                    std=self.cond_std,

                    columns=np.asarray(self.cond_columns),

                )



    @property

    def cond_dim(self):

        return len(self.cond_columns)



    def __len__(self):

        return len(self.meta)



    def __getitem__(self, idx):

        row = self.meta.iloc[idx]

        raw_cond = row[self.cond_columns].values.astype(np.float32)

        cond_tensor = torch.from_numpy((raw_cond - self.cond_mean) / self.cond_std)



        filepath = os.path.join(self.npy_dir, f"200yuan-{int(row['sim_id'])}.npy")

        if not os.path.exists(filepath):

            raise FileNotFoundError(filepath)

        img_array = np.load(filepath).astype(np.float32)

        if img_array.shape != (5, 64, 256):

            raise ValueError(f"Unexpected field shape {img_array.shape} in {filepath}")

        img_array = (img_array - self.mean_img[:, None, None]) / self.std_img[:, None, None]

        return cond_tensor, torch.from_numpy(img_array)

