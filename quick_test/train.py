import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

DATA_CSV = Path("dataset_v2.csv")
LABEL_JSON = Path("label_order.json")
QUEST_JSON = Path("questions_v2.json")
ONNX_PATH = Path("survey_v2.onnx")
SEED = 20250902
BS = 256
LR = 2e-3
WD = 1e-5
EPOCHS = 80
PATIENCE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class DS(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = df[[f"Q{i}" for i in range(1, 11)]].values.astype(np.float32)
        self.y = df[
            [c for c in df.columns if c.startswith("Q") == False]
        ].values.astype(np.float32)
        r = self.y.sum(axis=1, keepdims=True)
        self.y = self.y / np.clip(r, 1e-8, None)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class LikertCalibrator(nn.Module):
    def __init__(self, n_features=10):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(n_features, 1))
        self.delta = nn.Parameter(torch.zeros(n_features, 4))

    def forward(self, x):
        idx = x.long() - 1
        d = torch.nn.functional.softplus(self.delta)
        v = torch.cat([self.base, self.base + torch.cumsum(d, dim=1)], dim=1)
        idx = torch.clamp(idx, 0, 4)
        b = (
            torch.arange(v.size(0), device=v.device)
            .unsqueeze(1)
            .expand(-1, idx.size(0))
            .t()
        )
        return v[b, idx]


class MonotoneMultiLabel(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.cal = LikertCalibrator(10)
        self.w_raw = nn.Parameter(torch.zeros(n_labels, 10))
        self.b = nn.Parameter(torch.zeros(n_labels))

    def forward(self, x):
        z = self.cal(x)
        W = torch.nn.functional.softplus(self.w_raw)
        logits = z @ W.t() + self.b
        return torch.nn.functional.softmax(logits, dim=1)


def train():
    set_seed(SEED)
    labels = json.loads(LABEL_JSON.read_text(encoding="utf-8"))
    ds = DS(DATA_CSV)
    n = len(ds)
    n_tr = int(n * 0.85)
    tr, va = torch.utils.data.random_split(
        ds, [n_tr, n - n_tr], generator=torch.Generator().manual_seed(SEED)
    )
    tl = torch.utils.data.DataLoader(tr, batch_size=BS, shuffle=True, drop_last=False)
    vl = torch.utils.data.DataLoader(va, batch_size=BS, shuffle=False, drop_last=False)
    model = MonotoneMultiLabel(len(labels)).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best = 1e9
    wait = 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for X, y in tl:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            p = model(X)
            loss = torch.nn.functional.kl_div(
                torch.log(torch.clamp(p, 1e-8, 1.0)), y, reduction="batchmean"
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        vloss = 0.0
        c = 0
        with torch.no_grad():
            for X, y in vl:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                p = model(X)
                vloss += torch.nn.functional.kl_div(
                    torch.log(torch.clamp(p, 1e-8, 1.0)), y, reduction="batchmean"
                ).item()
                c += 1
        vloss /= max(1, c)
        if vloss < best - 1e-5:
            best = vloss
            wait = 0
            torch.save(model.state_dict(), "best_v2.pt")
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(torch.load("best_v2.pt", map_location=DEVICE))
    model.eval()
    dummy = torch.ones(1, 10, dtype=torch.float32).to(DEVICE)
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH.as_posix(),
        input_names=["likert10"],
        output_names=["probs"],
        dynamic_axes={"likert10": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=17,
    )


if __name__ == "__main__":
    if not DATA_CSV.exists():
        raise SystemExit("dataset_v2.csv 가 필요합니다.")
    if not LABEL_JSON.exists():
        raise SystemExit("label_order.json 가 필요합니다.")
    train()
