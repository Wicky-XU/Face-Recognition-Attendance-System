import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
from preprocess import LFW_Dataset, LFW_DATASET_PATH, transform

def train():
    # 确保 PyTorch 运行在 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 LFW 数据集
    lfw_dataset = LFW_Dataset(root_dir=LFW_DATASET_PATH, transform=transform)
    lfw_loader = DataLoader(lfw_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    # 加载预训练 FaceNet
    model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=len(set(lfw_dataset.labels))).to(device)

    # 冻结大部分层，仅微调最后几层
    for param in model.parameters():
        param.requires_grad = False
    for param in model.logits.parameters():
        param.requires_grad = True  # 仅训练最后的分类层

    # 修正 FutureWarning
    scaler = torch.amp.GradScaler("cuda")

    # **修改学习率**
    learning_rate = 1e-4
    optimizer = optim.Adam(model.logits.parameters(), lr=learning_rate)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    num_epochs = 20  # 训练 20 轮
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0  # 计算当前 epoch 的总 loss

        for images, labels in lfw_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):  # 修正 torch.cuda.amp.autocast()
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()  # 仅累加 loss，不打印 batch 级别 loss

        # ** 打印 epoch 结束时的 loss**
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(lfw_loader):.4f}")

    # 保存模型
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/facenet_lfw.pth")
    print("✅ Model saved as models/facenet_lfw.pth")


    # 只在主进程打印设备信息
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    train()