import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型并删除分类层
model = InceptionResnetV1(pretrained=None).to(device)
model_path = "C:/Users/29873/PycharmProjects/face_recognition_training/scripts/models/facenet_lfw.pth"
state_dict = torch.load(model_path, map_location=device)

# 删除 logits 层权重（分类头）
for key in list(state_dict.keys()):
    if "logits" in key:
        del state_dict[key]

model.load_state_dict(state_dict, strict=False)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 指定图片路径
img_path = "C:/Users/29873/Desktop/wicky.jpg"
img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

# 提取特征向量
with torch.no_grad():
    feature_vector = model(img)

print("Feature shape:", feature_vector.shape)
print("Feature vector:", feature_vector)
