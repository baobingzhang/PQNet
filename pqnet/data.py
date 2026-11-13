"""Benchmark loading with the splits, scaling and settings of the official TQNet scripts.

Every split is returned as a (T, C) float32 array together with the absolute index of
its first row in the full series, so that phases (tau mod P) are consistent across
train, validation and test.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch

# name: (reader, file, channels, hand-set cycle W, batch, lr, dropout, horizons)
DATASETS = {
    "ETTh1": ("ett_hour", "ETTh1.csv", 7, 24, 256, 1e-3, 0.5, [96, 192, 336, 720]),
    "ETTh2": ("ett_hour", "ETTh2.csv", 7, 24, 256, 1e-3, 0.5, [96, 192, 336, 720]),
    "ETTm1": ("ett_minute", "ETTm1.csv", 7, 96, 256, 1e-3, 0.5, [96, 192, 336, 720]),
    "ETTm2": ("ett_minute", "ETTm2.csv", 7, 96, 256, 1e-3, 0.5, [96, 192, 336, 720]),
    "Weather": ("custom", "weather.csv", 21, 144, 64, 1e-3, 0.5, [96, 192, 336, 720]),
    "Electricity": (
        "custom",
        "electricity.csv",
        321,
        168,
        32,
        3e-3,
        0.0,
        [96, 192, 336, 720],
    ),
    "Traffic": ("custom", "traffic.csv", 862, 168, 16, 3e-3, 0.0, [96, 192, 336, 720]),
    "Solar": ("solar", "solar_AL.txt", 137, 144, 64, 3e-3, 0.0, [96, 192, 336, 720]),
    "PEMS03": ("pems", "PEMS03.npz", 358, 288, 32, 3e-3, 0.0, [12, 24, 48, 96]),
    "PEMS04": ("pems", "PEMS04.npz", 307, 288, 32, 3e-3, 0.0, [12, 24, 48, 96]),
    "PEMS07": ("pems", "PEMS07.npz", 883, 288, 32, 3e-3, 0.0, [12, 24, 48, 96]),
    "PEMS08": ("pems", "PEMS08.npz", 170, 288, 32, 3e-3, 0.0, [12, 24, 48, 96]),
    "Exchange": (
        "custom",
        "exchange_rate.csv",
        8,
        1,
        32,
        1e-3,
        0.5,
        [96, 192, 336, 720],
    ),
    "ILI": ("custom", "national_illness.csv", 7, 52, 32, 1e-3, 0.5, [24, 36, 48, 60]),
}
# The official TQNet scripts disable RevIN on these datasets.
REVIN_OFF = {"PEMS03", "PEMS04", "PEMS07", "Solar"}


@dataclass
class DatasetConfig:
    name: str
    channels: int
    hand_cycle: int
    batch_size: int
    lr: float
    dropout: float
    horizons: list
    use_revin: bool


def config(name):
    _, _, c, w, bs, lr, do, hz = DATASETS[name]
    return DatasetConfig(name, c, w, bs, lr, do, hz, name not in REVIN_OFF)


def _read(name, root):
    import pandas as pd

    reader, fname = DATASETS[name][:2]
    path = os.path.join(root, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; see 'Data' in README.md")
    if reader == "pems":
        return np.load(path, allow_pickle=True)["data"][:, :, 0].astype(np.float64)
    if reader == "solar":
        return np.loadtxt(path, delimiter=",", dtype=np.float64)
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c not in ("date", "OT")] + [
        "OT"
    ]  # target column last
    return df[cols].values.astype(np.float64)


def _borders(reader, n, L):
    if reader == "ett_hour":
        m = 30 * 24
        return [0, 12 * m - L, 16 * m - L], [12 * m, 16 * m, 20 * m]
    if reader == "ett_minute":
        m = 30 * 24 * 4
        return [0, 12 * m - L, 16 * m - L], [12 * m, 16 * m, 20 * m]
    if reader == "pems":
        tr, te = int(n * 0.6), int(n * 0.2)
        va = int(n * 0.2)
    else:
        tr, te = int(n * 0.7), int(n * 0.2)
        va = int(n * 0.1) if reader == "solar" else n - tr - te
    return [0, tr - L, n - te - L], [tr, tr + va, n]


def load_splits(name, root, seq_len=96):
    """Return {'train'|'val'|'test': (array (T, C) float32, absolute start index)}."""
    data = _read(name, root)
    b1, b2 = _borders(DATASETS[name][0], len(data), seq_len)
    train = data[b1[0] : b2[0]]
    mean, std = train.mean(0), train.std(0)
    std[std == 0] = 1.0  # as sklearn.preprocessing.StandardScaler
    data = (data - mean) / std
    return {
        k: (data[b1[i] : b2[i]].astype(np.float32), b1[i])
        for i, k in enumerate(("train", "val", "test"))
    }


class WindowLoader:
    """Sliding windows (input L, target H) with the absolute index of the first target step.
    Same windows, shuffling and drop_last behaviour as the TQNet DataLoader."""

    def __init__(self, array, start, L, H, batch_size, shuffle, device="cpu"):
        self.x = torch.as_tensor(array, device=device)
        self.start, self.L, self.H, self.bs = start, L, H, batch_size
        self.shuffle = self.drop_last = shuffle
        self.n = len(array) - L - H + 1
        self.aL = torch.arange(L, device=device).view(1, -1)
        self.aH = torch.arange(H, device=device).view(1, -1) + L

    def __len__(self):
        return self.n // self.bs if self.drop_last else -(-self.n // self.bs)

    def __iter__(self):
        order = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for i in range(0, self.n, self.bs):
            if self.drop_last and i + self.bs > self.n:
                break
            idx = order[i : i + self.bs].to(self.x.device).view(-1, 1)
            t_end = self.start + idx.view(-1) + self.L
            yield self.x[idx + self.aL], self.x[idx + self.aH], t_end
