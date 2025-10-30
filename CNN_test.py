# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter

# =========================
# 1️⃣ Dataset
# =========================
class SpinalDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, label_file, subsets, transform=None):
        self.img_root = img_root
        self.subsets = subsets
        self.transform = transform

        self.df = pd.read_csv(label_file, header=None)
        self.df.columns = ["filename", "angle1", "angle2", "angle3"]

        self.file_map = {}
        for sub in subsets:
            folder = os.path.join(img_root, sub)
            for fname in os.listdir(folder):
                self.file_map[fname] = os.path.join(folder, fname)

        self.df = self.df[self.df["filename"].isin(self.file_map.keys())].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.file_map[row["filename"]]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor([row["angle1"], row["angle2"], row["angle3"]], dtype=torch.float32)
        return img, label


# =========================
# 2️⃣ CNN
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


# =========================
# 3️⃣ Cobb
# =========================
def cobb_to_class(angles):
    """angles: Tensor [B,3] -> 返回每张图的分类标签 [B]"""
    max_angle = torch.max(angles, dim=1)[0]
    classes = torch.zeros_like(max_angle, dtype=torch.long)
    classes[(max_angle >= 10) & (max_angle < 25)] = 1
    classes[(max_angle >= 25) & (max_angle < 45)] = 2
    classes[(max_angle >= 45)] = 3
    return classes


# =========================
# 4️⃣ test
# =========================
if __name__ == "__main__":
    data_root = "Spinal-AI2024"
    test_label_file = os.path.join(data_root, "Cobb_spinal-AI2024-test_gt.txt")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = SpinalDataset(
        img_root=data_root,
        label_file=test_label_file,
        subsets=["Spinal-AI2024-subset5"],
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("checkpoints/final_model.pth", map_location=device))
    model.eval()

    preds, trues = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            pred_classes = cobb_to_class(outputs.cpu())
            true_classes = cobb_to_class(labels)
            preds.extend(pred_classes.numpy())
            trues.extend(true_classes.numpy())

    # =========================
    # 5️⃣ ACC
    # =========================
    acc = accuracy_score(trues, preds)
    print(f"✅ Acc (Normal/Mild/Moderate/Severe): {acc * 100:.2f}%")

    print("Prediction Distribution:", Counter(preds))
    print("real Class Distribution:", Counter(trues))

    # Matrix
    cm = confusion_matrix(trues, preds, labels=[0, 1, 2, 3])
    plt.figure(figsize=(6, 5))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (SimpleCNN)")
    plt.tight_layout()
    plt.savefig("cnn_confusion_matrix.png", dpi=300)
    plt.show()

