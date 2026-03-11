import torch
import torch.nn as nn
import torch.optim as optim
from unet import UNet1D
from datasets import CSIDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

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
    model = UNet1D(num_encoding_blocks=3, in_channels=52, out_classes=1, out_channels_first_layer=64, preactivation=True, residual=True).to(device)
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
            # if (batch_idx + 1) % 10 == 0:
            #     print(f"Epoch [{epoch + 1}/{epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] | 当前 Loss: {loss.item():.4f}")

        print(f'epoch:{epoch}, epoch_loss: {epoch_loss:.4f}')
    # ================= 激动人心的预测与可视化环节 =================
    print("\n🎨 开始绘制预测结果...")
    model.eval()  # 极其关键：告诉大厨现在是考试，不要再修改脑细胞了

    # 1. 抽查第一个样本 (注意解包 x 和 真实的 y)
    # (假设你的 Dataset 变量名叫 train_dataset，如果你用的是 train_Dataset 请对应修改)
    x_sample, y_true_sample = train_Dataset[40]

    # 2. 升维与上显卡
    # 把 [52, 256] 变成 [1, 52, 256]，并送进 GPU
    x_input = x_sample.unsqueeze(0).to(device)

    # 3. 让模型进行预测
    with torch.no_grad():
        raw_output = model(x_input)
        # 🌟 核心魔法：把模型输出压扁到 0-1 之间！
        y_pred = torch.sigmoid(raw_output)

        # 4. 把数据从 GPU 拔下来，转换成 matplotlib 能看懂的 1D numpy 数组
    # .squeeze() 会把 [1, 1, 256] 极其丝滑地压成一根面条 [256]
    y_pred_np = y_pred.squeeze().cpu().numpy()
    y_true_np = y_true_sample.squeeze().cpu().numpy()

    # 5. 画在同一张图上，真假美猴王大比拼！
    plt.figure(figsize=(14, 6))

    # 画真实的 ECG 热力图（蓝色粗线）
    plt.plot(y_true_np, label="True ECG Heatmap (Ground Truth)", color="blue", linewidth=2.5, alpha=0.7)

    # 画模型预测的 热力图（红色虚线）
    plt.plot(y_pred_np, label="U-Net Predicted Heatmap", color="red", linestyle="--", linewidth=2.5)

    plt.title("Epoch 50: CSI-based Heartbeat Prediction vs Ground Truth", fontsize=16)
    plt.xlabel("Time Steps (Sliding Window)", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.show()

if __name__ == "__main__":
    train()