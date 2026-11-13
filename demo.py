"""Self-contained demo (no download, CPU, about one minute).

Builds a synthetic 8-sensor hourly stream with a daily and a weekly cycle, a slow trend and
noise, then
  1. discovers the period hierarchy on the training split (no sampling rate is given),
  2. trains PQNet on each nested prefix and lets the validation loss choose,
  3. compares with TQNet (hand-set daily cycle, no residual memory),
  4. saves demo_forecast.png.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

from pqnet import PQNet, discover_periods
from pqnet.data import WindowLoader


def synthetic_sensors(T=24 * 7 * 60, C=8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)[:, None]
    daily = np.sin(2 * np.pi * t / 24 + rng.uniform(0, 2 * np.pi, C))
    weekday = (t // 24) % 7 < 5  # weekday/weekend pattern, a weekly cycle
    weekly = (
        0.8
        * weekday
        * np.clip(np.sin(2 * np.pi * (t % 24) / 24), 0, None)
        * rng.uniform(0.5, 1.5, C)
    )
    trend = 0.5 * t / T * rng.normal(size=C)
    x = daily + weekly + trend + 0.3 * rng.normal(size=(T, C))
    return x.astype(np.float32)


def splits(x, L):
    n = len(x)
    tr, te = int(0.7 * n), int(0.2 * n)
    b1, b2 = [0, tr - L, n - te - L], [tr, n - te, n]
    mu, sd = x[:tr].mean(0), x[:tr].std(0)
    x = (x - mu) / sd
    return {
        k: (x[b1[i] : b2[i]], b1[i]) for i, k in enumerate(("train", "val", "test"))
    }


def fit(sp, periods, residual, L, H, epochs, seed=0):
    torch.manual_seed(seed)
    C = sp["train"][0].shape[1]
    model = PQNet(L, H, C, periods, d_model=128, residual=residual)
    tr = WindowLoader(*sp["train"], L, H, 64, shuffle=True)
    va = WindowLoader(*sp["val"], L, H, 256, shuffle=False)
    te = WindowLoader(*sp["test"], L, H, 256, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def mse(loader):
        model.eval()
        with torch.no_grad():
            v = np.mean([((model(x, t) - y) ** 2).mean().item() for x, y, t in loader])
        model.train()
        return float(v)

    best, state = None, None
    for _ in range(epochs):
        for x, y, t in tr:
            opt.zero_grad()
            nn.functional.mse_loss(model(x, t), y).backward()
            opt.step()
        v = mse(va)
        if best is None or v < best:
            best, state = v, {k: p.clone() for k, p in model.state_dict().items()}
    model.load_state_dict(state)
    return model, best, mse(te), te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--no_plot", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    L, H = 96, 48

    sp = splits(synthetic_sensors(), L)
    hierarchy, _ = discover_periods(sp["train"][0])
    print(f"1. discovered period hierarchy (training split only): {hierarchy}")

    print("2. PQNet, one model per nested prefix; validation picks the prefix")
    runs = []
    for k in range(1, len(hierarchy) + 1):
        m, val, test, te = fit(sp, hierarchy[:k], True, L, H, a.epochs)
        runs.append((val, test, hierarchy[:k], m, te))
        print(
            f"   periods {str(hierarchy[:k]):>10}  val MSE {val:.4f}  test MSE {test:.4f}"
        )
    val, test, chosen, model, te = min(runs, key=lambda r: r[0])

    _, _, tq_test, _ = fit(sp, [24], False, L, H, a.epochs)
    print(f"3. test MSE  PQNet {chosen}: {test:.4f}   TQNet [24]: {tq_test:.4f}")

    if not a.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x, y, t = next(iter(te))
        with torch.no_grad():
            p = model(x[:1], t[:1])[0, :, 0].numpy()
        fig, ax = plt.subplots(figsize=(7, 2.4))
        ax.plot(range(L), x[0, :, 0], color="0.3", lw=1, label="history")
        ax.plot(
            range(L, L + H),
            y[0, :, 0],
            color="0.3",
            lw=1,
            ls="--",
            label="ground truth",
        )
        ax.plot(range(L, L + H), p, color="#2a6fc4", lw=1.6, label=f"PQNet {chosen}")
        ax.axvline(L, color="0.7", lw=0.8)
        ax.set_xlabel("time step (hours)")
        ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig("demo_forecast.png", dpi=160)
        print("4. saved demo_forecast.png")


if __name__ == "__main__":
    main()
