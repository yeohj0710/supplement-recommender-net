import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast


class MLP(nn.Module):
    def __init__(self, in_dim=10, out_dim=22, hidden=128, depth=2, dropout=0.1):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            dim = hidden
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_data(path):
    d = np.load(os.path.join(path, "dataset.npz"), allow_pickle=True)
    X_train = d["X_train"].astype(np.float32)
    X_val = d["X_val"].astype(np.float32)
    X_test = d["X_test"].astype(np.float32)
    y_train = d["y_train"].astype(np.float32)
    y_val = d["y_val"].astype(np.float32)
    y_test = d["y_test"].astype(np.float32)
    cats = d["categories"].tolist()
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), cats


def to_tensor_dataset(X, y):
    Z = (X - 1.0) / 4.0
    return TensorDataset(torch.from_numpy(Z), torch.from_numpy(y))


def eval_uniform(model, device, cats):
    def pred_of(v):
        x = torch.from_numpy(v.astype(np.float32))
        z = (x - 1.0) / 4.0
        with torch.no_grad():
            p = model(z.to(device)).sigmoid().mean(0).cpu().numpy()
        return p

    a1 = np.ones((64, 10), dtype=np.float32)
    a3 = np.full((64, 10), 3, dtype=np.float32)
    a5 = np.full((64, 10), 5, dtype=np.float32)
    p1, p3, p5 = pred_of(a1), pred_of(a3), pred_of(a5)
    safe_idx = [
        cats.index(c)
        for c in [
            "종합비타민",
            "비타민C",
            "오메가3",
            "프로바이오틱스(유산균)",
            "비타민D",
        ]
    ]
    safe_mean_3 = float(p3[safe_idx].mean())
    all_mean_1 = float(p1.mean())
    all_mean_5 = float(p5.mean())
    return {
        "mean_all_1": all_mean_1,
        "safe_mean_all_3": safe_mean_3,
        "mean_all_5": all_mean_5,
    }


def train_one_epoch(model, loader, opt, scaler, device, loss_fn, grad_clip):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        with autocast():
            logits = model(xb)
            pred = torch.sigmoid(logits)
            loss = loss_fn(pred, yb)
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        total += float(loss.detach().cpu()) * xb.size(0)
    return total / len(loader.dataset)


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = torch.sigmoid(model(xb))
            loss = loss_fn(pred, yb)
            total += float(loss.detach().cpu()) * xb.size(0)
    return total / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"])
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    (Xtr, ytr), (Xva, yva), (Xte, yte), cats = load_data(args.data_dir)
    train_ds = to_tensor_dataset(Xtr, ytr)
    val_ds = to_tensor_dataset(Xva, yva)
    test_ds = to_tensor_dataset(Xte, yte)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(
        in_dim=10,
        out_dim=len(cats),
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, verbose=False
    )
    scaler = GradScaler()
    loss_fn = nn.SmoothL1Loss() if args.loss == "huber" else nn.MSELoss()

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "model.pt")
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    meta_path = os.path.join(args.out_dir, "meta.json")
    patience = args.patience
    hist = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, opt, scaler, device, loss_fn, args.grad_clip
        )
        val_loss = evaluate(model, val_loader, device, loss_fn)
        scheduler.step(val_loss)
        anchors = eval_uniform(model, device, cats)
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": opt.param_groups[0]["lr"],
            **anchors,
        }
        hist.append(rec)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience = args.patience
            torch.save(
                {"state_dict": model.state_dict(), "cats": cats, "config": vars(args)},
                best_path,
            )
        else:
            patience -= 1
            if patience <= 0:
                break

    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    test_loss = evaluate(model, test_loader, device, loss_fn)
    anchors = eval_uniform(model, device, cats)

    meta = {
        "categories": cats,
        "input_scale": {"type": "linear", "min_raw": 1.0, "max_raw": 5.0},
        "train_config": vars(args),
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "anchor_eval": anchors,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    dummy = torch.zeros(1, 10, dtype=torch.float32).to(device)
    torch.onnx.export(
        model,
        dummy,
        os.path.join(args.out_dir, "model.onnx"),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


if __name__ == "__main__":
    main()
