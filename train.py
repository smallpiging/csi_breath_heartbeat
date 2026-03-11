import torch
import torch.nn as nn
import torch.optim as optim
from unet import UNet1D
from datasets import CSIDataset
from torch.utils.data import DataLoader

def train():
    # 参数
    batch_size = 4
    learning_rate = 0.0001
    epochs = 50

    # 准备设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 准备数据集
    train_Dataset = CSIDataset(data_dir="./processed_datasets", window_size=256, step=20)
    train_loader = DataLoader(dataset=train_Dataset, batch_size=batch_size, shuffle=True)
    print(f'数据集加载成功，{len(train_Dataset)}个样本,分为{len(train_loader)}个批次')

    # 准备模型
    model = UNet1D(num_encoding_blocks=3, in_channels=52, out_classes=1, out_channels_first_layer=16, preactivation=True, residual=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print('模型参数：',total_params)

    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('开始训练')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            prediction = model(x_batch) # 前向传播
            loss = criterion(prediction, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] | 当前 Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()