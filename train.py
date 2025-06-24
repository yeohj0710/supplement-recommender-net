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
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
HIDDEN_DIM = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SurveyDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = df[FEATURE_COLUMNS].values.astype(np.float32)
        self.y = df[LABEL_COLUMNS].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = self.block1(x)
        residual = x
        x = self.block2(x)
        x = x + residual
        x = self.block3(x)
        return self.classifier(x)


def train():

    dataset = SurveyDataset(CSV_PATH)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = ImprovedMLP(len(FEATURE_COLUMNS), HIDDEN_DIM, len(LABEL_COLUMNS)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} - train_loss: {avg_train:.4f} - val_loss: {avg_val:.4f} - lr: {optimizer.param_groups[0]['lr']:.2e}"
        )

    dummy_input = torch.randn(1, len(FEATURE_COLUMNS)).to(DEVICE)
    torch.onnx.export(
        model,
        dummy_input,
        "survey_model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11,
    )
    print("ONNX model saved as survey_model.onnx")


if __name__ == "__main__":
    train()
