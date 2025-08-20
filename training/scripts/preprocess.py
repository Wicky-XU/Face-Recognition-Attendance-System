import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np

# 设置 LFW 数据集路径
LFW_DATASET_PATH = "../dataset/lfw-deepfunneled"

# 数据增强与预处理
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # 调整到 FaceNet 需要的 160x160
    transforms.RandomHorizontalFlip(),  # 随机水平翻转（数据增强）
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 亮度 & 对比度调整
    transforms.ToTensor(),  # 转换为 PyTorch Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
])

# 创建 PyTorch 数据集类
class LFW_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "*", "*.jpg"))  # 获取所有人脸图像路径
        self.labels = [os.path.basename(os.path.dirname(p)) for p in self.image_paths]  # 以文件夹名作为标签
        self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}  # 标签转换为索引

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")  # 打开图像
        label = self.label_to_idx[self.labels[idx]]  # 获取标签索引

        if self.transform:
            img = self.transform(img)

        return img, label

# 让 Windows 避免 multiprocessing 报错
if __name__ == "__main__":
    # 创建 LFW 数据加载器
    lfw_dataset = LFW_Dataset(root_dir=LFW_DATASET_PATH, transform=transform)
    lfw_loader = DataLoader(lfw_dataset, batch_size=32, shuffle=True, num_workers=0)  # num_workers=0

    print(f"✅ LFW 数据集加载完成，共 {len(lfw_dataset)} 张图片")

    # 获取一个 batch
    images, labels = next(iter(lfw_loader))

    # 反归一化函数（从 [-1, 1] 转换回 [0, 1]）
    def unnormalize(img):
        img = img.numpy().transpose(1, 2, 0)  # 调整通道顺序
        img = (img * 0.5) + 0.5  # 反归一化
        img = np.clip(img, 0, 1)  # 限制范围，防止超出 [0,1]
        return img

    # 显示前 5 张图片
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        axes[i].imshow(unnormalize(images[i]))  # 反归一化并显示
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis("off")

    plt.show()  # 显示图片
