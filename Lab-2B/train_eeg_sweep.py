# Full Lab 2B sweep, from ChatGPT
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, LeaveOneGroupOut

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Helpers
# =========================
def _subject_dirs(root: Path) -> list[Path]:
    """List subject directories named like S01 or (S01)."""
    subs = []
    for d in root.iterdir():
        if d.is_dir() and re.search(r"S\d{2}", d.name):
            subs.append(d)
    subs.sort(key=lambda p: int(re.search(r"S(\d{2})", p.name).group(1)))
    return subs

def _game_path(sub_dir: Path, sid: int, g: int) -> Optional[Path]:
    """Return CSV path for S{sid}G{g} under expected subtrees; None if missing."""
    fname = f"S{sid:02d}G{g}AllChannels.csv"
    p1 = sub_dir / "Preprocessed EEG Data" / ".csv format" / fname
    if p1.exists():
        return p1
    p2 = sub_dir / "Preprocessed" / ".csv format" / fname  # fallback seen in some copies
    if p2.exists():
        return p2
    return None

def _parse_fallback(s: str) -> list[str]:
    return [c.strip() for c in s.split(",") if c.strip()]

def _pick_channel(cols, primary: str, fallback: list[str]) -> tuple[Optional[str], bool]:
    """Pick primary if present; otherwise first available from fallback."""
    if primary in cols:
        return primary, False
    for ch in fallback:
        if ch in cols:
            return ch, True
    return None, False

def _parse_ids(s: str) -> set[int]:
    return set(int(x) for x in s.split(",") if x.strip())


# =========================
# Data loading
# =========================
def load_full_dataset(dataset_root: Path,
                      channel: str = "T7",
                      fs: int = 32,
                      clip_sec: int = 2,
                      channel_list: Optional[list[str]] = None,
                      exclude_subjects: Optional[set[int]] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, set[int]]:
    """
    Returns:
      X: (N, 1, T) float32   (T = fs * clip_sec, default 64)
      y: (N,) int64           labels in {0,1,2,3}
      groups: (N,) int64      subject ids (01..28)
      included_subjects: set[int] actually used
    """
    clip_len = fs * clip_sec
    X_list, y_list, g_list = [], [], []
    included_subjects = set()
    used_fallback_subjects = set()

    for sub_dir in _subject_dirs(dataset_root):
        sid = int(re.search(r"S(\d{2})", sub_dir.name).group(1))
        if exclude_subjects and sid in exclude_subjects:
            continue

        used_any = False
        for g in (1, 2, 3, 4):
            f = _game_path(sub_dir, sid, g)
            if f is None:
                continue
            df = pd.read_csv(f)
            # Drop any empty "Unnamed:*" columns
            df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

            ch, did_fb = _pick_channel(df.columns, channel, channel_list or [])
            if ch is None:
                continue

            x = df[ch].to_numpy()
            n = len(x) // clip_len
            if n == 0:
                continue
            x = x[: n * clip_len].reshape(n, clip_len).astype(np.float32)
            X_list.append(x)
            y_list.append(np.full(n, g - 1, dtype=np.int64))  # 0..3
            g_list.append(np.full(n, sid, dtype=np.int64))
            used_any = True
            if did_fb:
                used_fallback_subjects.add(sid)

        if used_any:
            included_subjects.add(sid)

    if not X_list:
        raise SystemExit(f"No usable CSV files found under: {dataset_root}")

    X = np.expand_dims(np.vstack(X_list), 1)  # (N,1,T)
    y = np.concatenate(y_list)
    groups = np.concatenate(g_list)

    if used_fallback_subjects:
        print("Used fallback channel(s) for subjects:", sorted(used_fallback_subjects))

    return X, y, groups, included_subjects


# =========================
# Model
# =========================
class SimpleCNN1D(nn.Module):
    """1D Conv stack that preserves temporal length (padding='same'); Flatten → Linear → LogSoftmax."""
    def __init__(self, in_ch: int, base_ch: int, depth: int, kernel: int, clip_len: int, n_classes: int = 4):
        super().__init__()
        layers = []
        c_in = in_ch
        for _ in range(depth):
            layers += [nn.Conv1d(c_in, base_ch, kernel_size=kernel, padding="same"), nn.ReLU()]
            c_in = base_ch
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_ch * clip_len, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# =========================
# Train / Eval
# =========================
@torch.no_grad()
def _acc_on(loader, model: nn.Module, device) -> float:
    model.eval()
    preds, targs = [], []
    for xb, yb in loader:
        pb = model(xb.to(device)).argmax(1).cpu().numpy()
        preds.append(pb)
        targs.append(yb.numpy())
    y_true = np.concatenate(targs)
    y_pred = np.concatenate(preds)
    return float((y_true == y_pred).mean())

def train_one(model: nn.Module,
              train_loader,
              test_loader,
              device,
              epochs: int = 30,
              lr: float = 1e-4) -> tuple[float, float, float, float]:
    """Returns: train_acc, test_acc, train_time_s, test_time_s"""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = nn.functional.nll_loss(model(xb), yb)
            loss.backward()
            opt.step()
    train_time = time.time() - t0

    t1 = time.time()
    test_acc = _acc_on(test_loader, model, device)
    test_time = time.time() - t1
    train_acc = _acc_on(train_loader, model, device)

    return train_acc, test_acc, train_time, test_time


# =========================
# Plotting
# =========================
def make_plots(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # Runtime vs params (train)
    plt.figure()
    plt.scatter(df["params_M"], df["train_time_s"])
    plt.xlabel("Model size (M params)")
    plt.ylabel("Train time (s)")
    plt.title("Runtime vs Model Size (train)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "runtime_vs_params_train.png", dpi=150)

    # Runtime vs params (test)
    plt.figure()
    plt.scatter(df["params_M"], df["test_time_s"])
    plt.xlabel("Model size (M params)")
    plt.ylabel("Test time (s)")
    plt.title("Runtime vs Model Size (test)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "runtime_vs_params_test.png", dpi=150)

    # Accuracy vs params
    plt.figure()
    plt.scatter(df["params_M"], df["test_acc"])
    plt.xlabel("Model size (M params)")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs Model Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "acc_vs_params.png", dpi=150)

    # Train time by config
    labels = [f"d{d}-c{c}" for d, c in zip(df["depth"], df["base_ch"])]
    xpos = np.arange(len(labels))
    plt.figure()
    plt.bar(xpos, df["train_time_s"])
    plt.xticks(xpos, labels, rotation=45, ha="right")
    plt.ylabel("Train time (s)")
    plt.title("Train time by configuration")
    plt.tight_layout()
    plt.savefig(outdir / "train_time_by_config.png", dpi=150)


# =========================
# Utilities
# =========================
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def build_loaders(X, y, test_size, seed, batch_size, cv_mode, groups=None):
    X_t = torch.tensor(X)                    # (N,1,T) float32
    y_t = torch.tensor(y, dtype=torch.long)

    if cv_mode == "random":
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_t.numpy(), y_t.numpy(),
            test_size=test_size, random_state=seed, stratify=y_t.numpy()
        )
        X_tr, X_te = torch.tensor(X_tr), torch.tensor(X_te)
        y_tr, y_te = torch.tensor(y_tr, dtype=torch.long), torch.tensor(y_te, dtype=torch.long)
        tr_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr, y_tr),
                                                batch_size=batch_size, shuffle=True)
        te_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_te, y_te),
                                                batch_size=batch_size, shuffle=False)
        return [(tr_loader, te_loader, None)]  # single split

    elif cv_mode == "logo":
        assert groups is not None, "groups (subject ids) required for LOGO"
        Xn, yn = X_t.numpy(), y_t.numpy()
        splits = []
        logo = LeaveOneGroupOut()
        for tr_idx, te_idx in logo.split(Xn, yn, groups):
            X_tr, X_te = torch.tensor(Xn[tr_idx]), torch.tensor(Xn[te_idx])
            y_tr, y_te = torch.tensor(yn[tr_idx], dtype=torch.long), torch.tensor(yn[te_idx], dtype=torch.long)
            tr_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr, y_tr),
                                                    batch_size=batch_size, shuffle=True)
            te_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_te, y_te),
                                                    batch_size=batch_size, shuffle=False)
            heldout_subj = int(np.unique(groups[te_idx])[0])
            splits.append((tr_loader, te_loader, heldout_subj))
        return splits

    else:
        raise ValueError(f"Unknown cv mode: {cv_mode}")

def sanity_print(X, y, groups, fs, clip_sec, included_subs: set[int]):
    print(f"X shape: {X.shape}  (N, C, T)   expect T={fs*clip_sec}")
    print(f"y shape: {y.shape}  labels: {sorted(set(y.tolist()))}  (expect [0,1,2,3])")
    subs = np.unique(groups)
    print(f"subjects in dataset: {subs.size}  (listed) → {sorted(included_subs)}")
    vals, cnts = np.unique(y, return_counts=True)
    print("label counts:", dict(zip(vals, cnts)))


# =========================
# Main
# =========================
def parse_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True, help="Path to GAMEEMO root .../GAMEEMO")
    ap.add_argument("--channel", type=str, default="T7", help="Primary channel to use")
    ap.add_argument("--channel-list", type=str, default="",
                    help="Fallback channels, comma-separated (e.g., 'AF3,FC6,AF4,FC5,....')")
    ap.add_argument("--exclude-subjects", type=str, default="", help="Subject IDs to skip, e.g., '26' or '13,26'")
    ap.add_argument("--fs", type=int, default=32)
    ap.add_argument("--clip-sec", type=int, default=2)
    ap.add_argument("--cv", type=str, default="random", choices=["random","logo"])
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--depths", type=str, default="1,2,3,2,3", help="comma-separated depths (≥5 entries)")
    ap.add_argument("--base-chs", type=str, default="8,8,8,16,16", help="comma-separated widths (paired with depths)")
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--outputs", type=str, default="outputs")
    ap.add_argument("--sanity", action="store_true", help="Print dataset stats and exit")
    ap.add_argument("--save-folds", action="store_true", help="When --cv logo, save per-fold CSV")
    args = ap.parse_args()

    outdir = Path(args.outputs)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root)
    fallback = _parse_fallback(args.channel_list)
    exclude = _parse_ids(args.exclude_subjects)

    X, y, groups, included_subs = load_full_dataset(
        dataset_root, channel=args.channel, fs=args.fs, clip_sec=args.clip_sec,
        channel_list=fallback, exclude_subjects=exclude
    )
    clip_len = X.shape[-1]

    if args.sanity:
        sanity_print(X, y, groups, args.fs, args.clip_sec, included_subs)
        return

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # Build splits/loaders
    splits = build_loaders(X, y, test_size=args.test_size, seed=args.seed,
                           batch_size=args.batch_size, cv_mode=args.cv, groups=groups)

    depths = parse_list(args.depths)
    base_chs = parse_list(args.base_chs)
    if len(depths) != len(base_chs):
        raise SystemExit("depths and base_chs must have equal length (paired configs).")

    rows = []
    rows_folds = []

    for depth, base_ch in zip(depths, base_chs):
        tr_accs, te_accs, tr_times, te_times = [], [], [], []
        for tr_loader, te_loader, heldout_subject in splits:
            model = SimpleCNN1D(in_ch=1, base_ch=base_ch, depth=depth,
                                kernel=args.kernel, clip_len=clip_len, n_classes=4)
            tr_acc, te_acc, tr_t, te_t = train_one(model, tr_loader, te_loader, device,
                                                   epochs=args.epochs, lr=args.lr)
            tr_accs.append(tr_acc); te_accs.append(te_acc)
            tr_times.append(tr_t);  te_times.append(te_t)

            if args.cv == "logo" and args.save-folds:
                rows_folds.append({
                    "heldout_subject": heldout_subject,
                    "depth": depth, "base_ch": base_ch,
                    "epochs": args.epochs, "batch_size": args.batch_size,
                    "train_acc": tr_acc, "test_acc": te_acc,
                    "train_time_s": tr_t, "test_time_s": te_t,
                    "params_M": count_params(model)/1e6,
                    "channel": args.channel
                })

        rows.append({
            "depth": depth, "base_ch": base_ch,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "clip_len": clip_len, "channel": args.channel, "cv": args.cv,
            "train_acc": float(np.mean(tr_accs)),
            "test_acc":  float(np.mean(te_accs)),
            "train_time_s": float(np.mean(tr_times)),
            "test_time_s":  float(np.mean(te_times)),
            "params_M": round(count_params(model)/1e6, 6)
        })
        print(f"[d={depth} c={base_ch}] train_acc={np.mean(tr_accs):.3f} "
              f"test_acc={np.mean(te_accs):.3f} train_s={np.mean(tr_times):.1f} "
              f"test_s={np.mean(te_times):.2f} params={count_params(model)/1e6:.3f}M")

    # Save sweep results
    df = pd.DataFrame(rows).sort_values(["depth","base_ch"])
    df.to_csv(outdir / "results_sweep.csv", index=False)

    if rows_folds:
        pd.DataFrame(rows_folds).to_csv(outdir / "results_folds.csv", index=False)

    with open(outdir / "summary.json", "w") as f:
        json.dump({
            "n_samples": int(X.shape[0]),
            "clip_len": int(clip_len),
            "n_subjects_included": len(included_subs),
            "subjects_included": sorted(int(s) for s in included_subs),
            "cv": args.cv,
            "channel_primary": args.channel,
            "channel_fallback": fallback,
            "configs": rows,
        }, f, indent=2)

    # Plots
    make_plots(df, outdir)
    print(f"\nSaved: {outdir/'results_sweep.csv'}  {outdir/'summary.json'}"
          f"{'  '+str(outdir/'results_folds.csv') if rows_folds else ''}"
          f" and PNG plots in {outdir}/")


if __name__ == "__main__":
    main()

