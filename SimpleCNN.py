# -*- coding: utf-8 -*-
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm  # ✅ 进度条
# =========================
# 1️⃣ 自定义 Dataset（自动从多个文件夹查找图像）
# =========================
class SpinalDataset(Dataset):
    def __init__(self, img_root, label_file, subsets, transform=None):
        """
        img_root: 数据集根目录，如 "Spinal-AI2024"
        label_file: 标注文件路径
        subsets: 使用的子文件夹列表，例如 ["subset1", "subset2", "subset3", "subset4"]
        transform: torchvision transforms
        """
        self.img_root = img_root
        self.subsets = subsets
        self.transform = transform

        # 读取标注文件
        self.df = pd.read_csv(label_file, header=None)
        self.df.columns = ["filename", "angle1", "angle2", "angle3"]

        # 建立 filename -> 文件路径 映射
        self.file_map = {}
        for sub in subsets:
            folder = os.path.join(img_root, sub)
            for fname in os.listdir(folder):
                self.file_map[fname] = os.path.join(folder, fname)

        # 保留存在于 file_map 中的样本（防止标注比图片多或少）
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

#
# class ResNet(nn.Module):
#     def __init__(self):
#         super(ResNet, self).__init__()
#         self.backbone = models.resnet18(pretrained=True)
#         self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 3)
#
#     def forward(self, x):
#         return self.backbone(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),   # [B, 3, 128, 128] -> [B, 16, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(2),                 # -> [B, 16, 64, 64]

            nn.Conv2d(16, 32, 3, padding=1),  # -> [B, 32, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2),                 # -> [B, 32, 32, 32]

            nn.Conv2d(32, 64, 3, padding=1),  # -> [B, 64, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2),                 # -> [B, 64, 16, 16]

            nn.Conv2d(64, 128, 3, padding=1), # -> [B, 128, 16, 16]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))     # -> [B, 128, 4, 4]
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 回归3个角度
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


if __name__ == '__main__':
    # =========================
    # 2️⃣ 数据加载
    # =========================
    data_root = "Spinal-AI2024"  # 根目录名
    train_label_file = os.path.join(data_root, "Cobb_spinal-AI2024-train_gt.txt")
    test_label_file  = os.path.join(data_root, "Cobb_spinal-AI2024-test_gt.txt")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 自动合并 subset1-4
    train_dataset = SpinalDataset(
        img_root=data_root,
        label_file=train_label_file,
        subsets=["Spinal-AI2024-subset1", "Spinal-AI2024-subset2", "Spinal-AI2024-subset3", "Spinal-AI2024-subset4"],
        transform=transform
    )

    # 测试集用 subset5
    test_dataset = SpinalDataset(
        img_root=data_root,
        label_file=test_label_file,
        subsets=["Spinal-AI2024-subset5"],
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    print(f"✅ 训练样本数: {len(train_dataset)}")
    print(f"✅ 测试样本数: {len(test_dataset)}")

    # =========================
    # 3️⃣ CNN 模型定义（回归3个角度）
    # =========================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # =========================
    # 4️⃣ 训练循环
    # =========================

    epochs = 100
    train_losses = []  # 📊 记录每个 epoch 的 loss
    test_maes = []  # 📊 （可选）记录每个 epoch 的 MAE

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]")
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"📦 Epoch [{epoch + 1}/{epochs}] 平均训练损失: {avg_loss:.4f}")

        # ✅ 每个 epoch 测试一次 MAE（可选）
        model.eval()
        mae = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                mae += torch.mean(torch.abs(outputs - labels)).item()
        mae /= len(test_loader)
        test_maes.append(mae)
        print(f"📊 测试集 MAE: {mae:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/final_model.pth")
    print("✅ 模型已保存到 checkpoints/final_model.pth")

    # 保存训练曲线到 CSV
    loss_df = pd.DataFrame({
        "epoch": list(range(1, epochs+1)),
        "train_loss": train_losses,
        "test_mae": test_maes
    })
    loss_df.to_csv("training_log.csv", index=False, encoding="utf-8-sig")
    print("📊 训练日志已保存到 training_log.csv")