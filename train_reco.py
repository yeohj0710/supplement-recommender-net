import argparse, json, os, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class UserTower(nn.Module):
    def __init__(self, in_dim: int, z_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        n = torch.norm(z, dim=1, keepdim=True) + 1e-8
        return z / n


class Head(nn.Module):
    def __init__(self, z_dim: int, num_labels: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(num_labels, z_dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(num_labels))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z
        logits = z @ self.w.t() + self.b
        return logits


def l2_normalize_rows(weight: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(weight, axis=1, keepdims=True) + 1e-8
    return weight / n


def build_feature_spec(df_x: pd.DataFrame):
    spec = []
    for c in df_x.columns:
        v = df_x[c].astype(float).values
        mu = float(np.mean(v))
        sigma = float(np.std(v) + 1e-8)
        spec.append(
            {"name": c, "kind": "num", "pool": "mean", "mu": mu, "sigma": sigma}
        )
    return {"factors": spec}


def standardize(df_x: pd.DataFrame, spec):
    arr = []
    for s in spec["factors"]:
        v = df_x[s["name"]].astype(float).values
        a = (v - s["mu"]) / s["sigma"]
        arr.append(a.reshape(-1, 1))
    X = np.concatenate(arr, axis=1).astype("float32")
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X = X / n
    return X


def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def compute_pos_weight(y: np.ndarray):
    n, c = y.shape
    pw = []
    for j in range(c):
        p = float(y[:, j].sum())
        nneg = n - p
        if p <= 0:
            pw.append(1.0)
        else:
            pw.append(max(1.0, nneg / p))
    return torch.tensor(pw, dtype=torch.float32)


def eval_metrics(model, head, X, y):
    model.eval()
    with torch.no_grad():
        z = model(torch.from_numpy(X))
        logits = head(z).numpy()
        p = 1.0 / (1.0 + np.exp(-logits))
        yb = (p >= 0.5).astype(np.float32)
        acc = float((yb == y).mean())
        return acc, logits


def fit_platt(logits: np.ndarray, y: np.ndarray):
    x = logits.reshape(-1, 1)
    t = y.reshape(-1)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x, t)
    a = float(clf.coef_.ravel()[0])
    b = float(clf.intercept_.ravel()[0])
    return {"method": "platt", "a": a, "b": b}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--feature-prefix", type=str, default="f_")
    p.add_argument("--label-prefix", type=str, default="y_")
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="./artifacts")
    args = p.parse_args()

    seed_all(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data)
    feat_cols = [c for c in df.columns if c.startswith(args.feature_prefix)]
    label_cols = [c for c in df.columns if c.startswith(args.label_prefix)]
    if len(feat_cols) == 0 or len(label_cols) == 0:
        raise RuntimeError("invalid columns")

    df_x = df[feat_cols].copy()
    df_y = df[label_cols].copy().astype(float).clip(0, 1)

    spec = build_feature_spec(df_x)
    with open(
        os.path.join(args.outdir, "feature_spec.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)

    X = standardize(df_x, spec)
    Y = df_y.values.astype("float32")

    Xtr, Xva, Ytr, Yva = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=args.seed,
        stratify=(Y.mean(axis=1) > 0).astype(int),
    )

    in_dim = X.shape[1]
    num_labels = Y.shape[1]
    model = UserTower(in_dim, args.z_dim, args.hidden)
    head = Head(args.z_dim, num_labels)
    pos_weight = compute_pos_weight(Ytr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=args.lr
    )
    best = None
    best_state = None
    patience = 8
    bad = 0

    for epoch in range(args.epochs):
        model.train()
        idx = np.random.permutation(len(Xtr))
        xb = torch.from_numpy(Xtr[idx])
        yb = torch.from_numpy(Ytr[idx])
        for i in range(0, len(idx), args.batch_size):
            x = xb[i : i + args.batch_size]
            y = yb[i : i + args.batch_size]
            z = model(x)
            logits = head(z)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc, logits_va = eval_metrics(model, head, Xva, Yva)
        score = acc
        if best is None or score > best:
            best = score
            best_state = {"model": model.state_dict(), "head": head.state_dict()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        head.load_state_dict(best_state["head"])

    _, logits_va = eval_metrics(model, head, Xva, Yva)
    calib = fit_platt(logits_va.reshape(-1, 1), Yva.reshape(-1, 1))
    with open(
        os.path.join(args.outdir, "calibration.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)

    with torch.no_grad():
        w = head.w.detach().cpu().numpy()
        b = head.b.detach().cpu().numpy()
    w = l2_normalize_rows(w)
    cat = {}
    for j, col in enumerate(label_cols):
        name = col[len(args.label_prefix) :]
        cat[name] = {"name": name, "w": w[j].tolist(), "b": float(b[j])}
    with open(os.path.join(args.outdir, "cat_params.json"), "w", encoding="utf-8") as f:
        json.dump(cat, f, ensure_ascii=False, indent=2)

    class ExportUser(nn.Module):
        def __init__(self, user_tower: UserTower):
            super().__init__()
            self.user = user_tower

        def forward(self, x):
            return self.user(x)

    export_model = ExportUser(model)
    export_in = torch.randn(1, in_dim, dtype=torch.float32)
    torch.onnx.export(
        export_model,
        export_in,
        os.path.join(args.outdir, "f_user.onnx"),
        input_names=["input"],
        output_names=["z"],
        dynamic_axes={"input": {0: "batch"}, "z": {0: "batch"}},
        opset_version=17,
    )

    meta = {
        "feature_prefix": args.feature_prefix,
        "label_prefix": args.label_prefix,
        "input_dim": in_dim,
        "z_dim": args.z_dim,
        "labels": [c[len(args.label_prefix) :] for c in label_cols],
    }
    with open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
