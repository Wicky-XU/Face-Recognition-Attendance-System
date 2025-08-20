import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = InceptionResnetV1(pretrained=None).to(device)
model_path = "C:/Users/29873/PycharmProjects/face_recognition_training/scripts/models/facenet_lfw.pth"
state_dict = torch.load(model_path, map_location=device)

for key in list(state_dict.keys()):
    if "logits" in key:
        del state_dict[key]

model.load_state_dict(state_dict, strict=False)
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(img)

# Compare two images
img1_path = "C:/Users/29873/Desktop/wicky.jpg"
img2_path = "C:/Users/29873/Desktop/wicky2.jpg"

embedding1 = get_embedding(img1_path)
embedding2 = get_embedding(img2_path)

cosine_similarity = F.cosine_similarity(embedding1, embedding2)
print(f"Cosine Similarity: {cosine_similarity.item()}")

# Decide whether it is the same person
threshold = 0.6
if cosine_similarity.item() > threshold:
    print("Same person")
else:
    print("Different person")
