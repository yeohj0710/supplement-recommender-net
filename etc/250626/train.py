import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

CSV_PATH = "survey_labels.csv"
FEATURE_COLUMNS = [f"Q{i}" for i in range(1, 11)]
LABEL_COLUMNS = [
    "비타민C",
    "칼슘",
    "마그네슘",
    "비타민D",
    "아연",
    "프로바이오틱스",
    "밀크씨슬",
    "오메가3",
    "멀티비타민",
    "차전자피 식이섬유",
    "철분",
    "엽산",
    "가르시니아",
    "콜라겐",
    "셀레늄",
    "루테인",
    "비타민A",
]
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200
HIDDEN_DIM1 = 256
HIDDEN_DIM2 = 128
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SurveyDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = df[FEATURE_COLUMNS].values.astype(np.float32)
        question_to_cats = {
            1: ["비타민C", "멀티비타민", "철분", "오메가3", "마그네슘"],
            2: ["칼슘", "비타민D", "마그네슘", "콜라겐"],
            3: ["마그네슘", "오메가3", "비타민D", "프로바이오틱스"],
            4: ["프로바이오틱스", "차전자피 식이섬유", "밀크씨슬"],
            5: ["비타민C", "아연", "셀레늄", "비타민D"],
            6: ["콜라겐", "비타민A", "비타민C", "셀레늄"],
            7: ["루테인", "오메가3", "비타민A"],
            8: ["엽산", "철분", "멀티비타민"],
            9: ["오메가3", "멀티비타민", "셀레늄"],
            10: ["비타민C", "셀레늄", "루테인", "밀크씨슬"],
        }
        label_idx = {lbl: i for i, lbl in enumerate(LABEL_COLUMNS)}
        n = len(self.X)
        self.y = np.zeros((n, len(LABEL_COLUMNS)), dtype=np.float32)
        for i in range(n):
            sums = np.zeros(len(LABEL_COLUMNS), dtype=np.float32)
            counts = np.zeros(len(LABEL_COLUMNS), dtype=np.int32)
            for q_idx in range(len(FEATURE_COLUMNS)):
                norm = (self.X[i, q_idx] - 1) / 4
                for cat in question_to_cats[q_idx + 1]:
                    j = label_idx[cat]
                    sums[j] += norm
                    counts[j] += 1
            self.y[i] = sums / (counts + 1e-8)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, h1, h2, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train():
    ds = SurveyDataset(CSV_PATH)
    n_train = int(len(ds) * 0.8)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds) - n_train])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = ImprovedMLP(
        len(FEATURE_COLUMNS), HIDDEN_DIM1, HIDDEN_DIM2, len(LABEL_COLUMNS)
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_val, no_imp = float("inf"), 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        tl = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            tl += loss.item()
        model.eval()
        mv = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                mv += criterion(model(X), y).item()
        avg_v = mv / len(val_loader)
        scheduler.step(avg_v)
        print(f"Epoch {epoch}/{NUM_EPOCHS} - val_loss: {avg_v:.4f}")
        if avg_v < best_val:
            best_val, no_imp = avg_v, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print("Early stopping")
                break

    dummy = torch.randn(1, len(FEATURE_COLUMNS)).to(DEVICE)
    torch.onnx.export(
        model,
        dummy,
        "survey_model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11,
    )


if __name__ == "__main__":
    train()
