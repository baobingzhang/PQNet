"""Train and evaluate PQNet (or the TQNet special case) on one benchmark and horizon.

Protocol of the official TQNet scripts: Adam, MSE loss, the per-dataset learning rate and
batch size, step decay (lr * 0.8^(epoch-3) from epoch 3), at most 30 epochs, early stopping
with patience 5 on the validation loss, best validation checkpoint evaluated on test.

Examples:
    python train.py --dataset ETTh1 --pred_len 96                 # PQNet (discovered periods, validation-selected)
    python train.py --dataset ETTh1 --pred_len 96 --model tqnet   # TQNet: hand-set cycle, no residual memory
    python train.py --dataset Electricity --pred_len 96 --periods 24 168
"""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from pqnet import PQNet, discover_periods
from pqnet.data import WindowLoader, config, load_splits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def val_loss(model, loader):
    """Validation loss: mean of the batch MSEs, as in the TQNet scripts."""
    model.eval()
    with torch.no_grad():
        losses = [((model(x, t) - y) ** 2).mean().item() for x, y, t in loader]
    model.train()
    return float(np.mean(losses))


def test_metrics(model, loader):
    """Test MSE and MAE over all windows and channels."""
    model.eval()
    se = ae = 0.0
    n = 0
    with torch.no_grad():
        for x, y, t in loader:
            err = model(x, t) - y
            se += (err**2).sum().item()
            ae += err.abs().sum().item()
            n += y.numel()
    model.train()
    return se / n, ae / n


def train_one(splits, cfg, periods, residual, args, device):
    set_seed(args.seed)
    L, H = args.seq_len, args.pred_len
    lr0 = args.lr or cfg.lr
    bs = args.batch_size or cfg.batch_size
    tr = WindowLoader(*splits["train"], L, H, bs, shuffle=True, device=device)
    va = WindowLoader(*splits["val"], L, H, bs, shuffle=False, device=device)
    te = WindowLoader(*splits["test"], L, H, bs, shuffle=False, device=device)

    model = PQNet(
        L,
        H,
        cfg.channels,
        periods,
        d_model=args.d_model,
        dropout=cfg.dropout,
        use_revin=cfg.use_revin,
        residual=residual,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr0)
    crit = nn.MSELoss()
    best, best_state, bad, t0 = None, None, 0, time.time()
    for ep in range(1, args.epochs + 1):
        tl = []
        for x, y, t_end in tr:
            opt.zero_grad()
            loss = crit(model(x, t_end), y)
            loss.backward()
            opt.step()
            tl.append(loss.item())
        v = val_loss(model, va)
        print(
            f"  periods {periods} epoch {ep:2d}  train {np.mean(tl):.4f}  val {v:.4f}  ({time.time() - t0:.0f}s)"
        )
        if best is None or v < best:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
        for g in opt.param_groups:
            g["lr"] = lr0 if ep < 3 else lr0 * 0.8 ** (ep - 3)
    model.load_state_dict(best_state)
    mse, mae = test_metrics(model, te)
    nparam = sum(p.numel() for p in model.parameters())
    return (
        dict(
            periods=periods,
            best_val=best,
            mse=mse,
            mae=mae,
            params=nparam,
            train_time_s=time.time() - t0,
        ),
        model,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--data_root", default="./dataset")
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--model", choices=["pqnet", "tqnet"], default="pqnet")
    ap.add_argument(
        "--periods",
        nargs="*",
        type=int,
        default=None,
        help="fixed periods; default: discovered on the training split and selected on validation",
    )
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument(
        "--lr", type=float, default=None, help="default: official per-dataset value"
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="default: official per-dataset value",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="optional JSON file for the result")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = config(args.dataset)
    splits = load_splits(args.dataset, args.data_root, args.seq_len)

    if args.model == "tqnet":
        candidates, residual = [args.periods or [cfg.hand_cycle]], False
    elif args.periods:
        candidates, residual = [args.periods], True
    else:
        hierarchy, _ = discover_periods(splits["train"][0], gamma=0.0)
        candidates, residual = [
            hierarchy[:k] for k in range(1, len(hierarchy) + 1)
        ], True
        print(f"discovered period hierarchy on the training split: {hierarchy}")

    runs = [train_one(splits, cfg, c, residual, args, device)[0] for c in candidates]
    res: dict = min(
        runs, key=lambda r: r["best_val"]
    )  # validation selects the prefix; test is not used
    res.update(
        dataset=args.dataset,
        model=args.model,
        pred_len=args.pred_len,
        seed=args.seed,
        candidates=[dict(periods=r["periods"], best_val=r["best_val"]) for r in runs],
    )
    print(
        f"{args.model} {args.dataset} H={args.pred_len} periods={res['periods']}: "
        f"test MSE {res['mse']:.4f}  MAE {res['mae']:.4f}  ({res['params']:,} parameters)"
    )
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
