import os, argparse, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class CSDataset(Dataset):
    def __init__(self, csv_path, cat_order):
        df = pd.read_csv(csv_path)
        self.cat_to_id = {k: i for i, k in enumerate(cat_order)}
        keep = ["cat_key", "v1", "v2", "v3", "v4", "v5", "label_0_100"]
        has_vcols = all([(f"v{i}" in df.columns) for i in range(1, 6)])
        if not has_vcols:
            vcols = []
            for i in range(1, 6):
                qids = [c for c in df.columns if c.endswith(f".{i}")]
                if len(qids) == 1:
                    vcols.append(qids[0])
                else:
                    raise RuntimeError("normalized v1..v5 not found")
            for i, src in enumerate(vcols, 1):
                col = df[src].astype(float)
                df[f"v{i}"] = np.where(
                    df[src].astype(str).isin(["0", "1"]), col, col / 3.0
                )
        self.df = df[["cat_key", "v1", "v2", "v3", "v4", "v5", "label_0_100"]].copy()
        self.df = self.df[self.df["cat_key"].isin(cat_order)].reset_index(drop=True)
        self.Xv = self.df[["v1", "v2", "v3", "v4", "v5"]].values.astype(np.float32)
        self.y = self.df["label_0_100"].values.astype(np.float32) / 100.0
        self.cid = self.df["cat_key"].map(self.cat_to_id).values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.cid[idx]),
            torch.tensor(self.Xv[idx]),
            torch.tensor(self.y[idx]),
        )


class CSNet(nn.Module):
    def __init__(self, num_cat=22, emb_dim=8, hidden=(64, 64), temperature=5.0):
        super().__init__()
        self.emb = nn.Embedding(num_cat, emb_dim)
        self.fc1 = nn.Linear(5 + emb_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], 1)
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.temp = torch.tensor(temperature, dtype=torch.float32)

    def forward(self, cat_ids, answers):
        e = self.emb(cat_ids)
        x = torch.cat([answers, e], dim=1)
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        z = self.out(h).squeeze(1)
        s = self.sig(z)
        p = torch.softmax(s * self.temp, dim=0)
        topv, topi = torch.topk(p, k=3, dim=0)
        return s, p, topv, topi


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def split_indices(n, val_ratio=0.15, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    v = int(n * val_ratio)
    return idx[v:], idx[:v]


def train_loop(model, train_loader, val_loader, epochs, lr, wd, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.L1Loss()
    best = None
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        n = 0
        for cid, xv, y in train_loader:
            cid, xv, y = cid.to(device), xv.to(device), y.to(device)
            s, _, _, _ = model(cid, xv)
            loss = loss_fn(s, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * y.size(0)
            n += y.size(0)
        model.eval()
        vl = 0.0
        m = 0
        with torch.no_grad():
            for cid, xv, y in val_loader:
                cid, xv, y = cid.to(device), xv.to(device), y.to(device)
                s, _, _, _ = model(cid, xv)
                loss = loss_fn(s, y)
                vl += loss.item() * y.size(0)
                m += y.size(0)
        sched.step()
        vloss = vl / max(m, 1)
        if best is None or vloss < best:
            best = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)


def export_onnx(model, out_path, cat_order):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.eval()
    with torch.no_grad():
        dummy_c = torch.tensor([0, 1, 2], dtype=torch.long)
        dummy_x = torch.rand(3, 5, dtype=torch.float32)
        torch.onnx.export(
            model,
            (dummy_c, dummy_x),
            out_path,
            input_names=["cat_ids", "answers"],
            output_names=["score_0_1", "percent_0_1", "topk_values", "topk_indices"],
            opset_version=17,
            dynamic_axes={
                "cat_ids": {0: "B"},
                "answers": {0: "B"},
                "score_0_1": {0: "B"},
                "percent_0_1": {0: "B"},
            },
            do_constant_folding=True,
        )
    with open(out_path.replace(".onnx", ".cats.json"), "w", encoding="utf-8") as f:
        json.dump({"cat_order": cat_order}, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", type=str, default="./data/c-section-v1/C-section-c-1.0-all.csv"
    )
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--embed", type=int, default=8)
    ap.add_argument("--hid1", type=int, default=64)
    ap.add_argument("--hid2", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="./model/c-section-scorer-v1.onnx")
    args = ap.parse_args()
    set_seed(args.seed)
    cat_order = [
        "vitc",
        "omega3",
        "ca",
        "lutein",
        "vitd",
        "milkthistle",
        "probiotics",
        "vitb",
        "mg",
        "garcinia",
        "multivitamin",
        "zn",
        "psyllium",
        "minerals",
        "vita",
        "fe",
        "ps",
        "folate",
        "arginine",
        "chondroitin",
        "coq10",
        "collagen",
    ]
    ds = CSDataset(args.csv, cat_order)
    idx_tr, idx_va = split_indices(len(ds), val_ratio=0.15, seed=args.seed)
    tr = torch.utils.data.Subset(ds, idx_tr)
    va = torch.utils.data.Subset(ds, idx_va)
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=False)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, drop_last=False)
    device = torch.device("cpu")
    model = CSNet(
        num_cat=len(cat_order),
        emb_dim=args.embed,
        hidden=(args.hid1, args.hid2),
        temperature=args.temperature,
    ).to(device)
    train_loop(model, dl_tr, dl_va, args.epochs, args.lr, args.wd, device)
    export_onnx(model, args.out, cat_order)


if __name__ == "__main__":
    main()
