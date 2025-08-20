import cv2
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

# Initialize the device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the cropping model
mtcnn = MTCNN().to(device)

# Load the FaceNet model
model = InceptionResnetV1(pretrained=None).to(device)
state_dict = torch.load("models/facenet_lfw.pth", map_location=device)

# Remove the weight of the classification layer
for key in list(state_dict.keys()):
    if "logits" in key:
        del state_dict[key]

model.load_state_dict(state_dict, strict=False)
model.eval()

# Extract features
def extract_face_encodings(image):
    # image is a BGR image in OpenCV format
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Face cropping
    aligned = mtcnn(img_pil)

    if aligned is None:
        return None

    aligned = aligned.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(aligned).squeeze(0).cpu().numpy()
    return emb
