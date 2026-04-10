import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import efficientnet_b0
import torch.nn as nn
from PIL import Image

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 43)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model

CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing >3.5t", "Right of way",
    "Priority road", "Yield", "Stop", "No vehicles", "No vehicles >3.5t",
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware ice/snow", "Wild animals", "End restrictions",
    "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left", "Roundabout",
    "End no passing", "End no passing >3.5t"
]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------- UI ----------
st.title("🚦 Traffic Sign Recognition")
st.write("Upload a traffic sign image and the model will classify it.")

uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", width=250)

    model = load_model()
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        top3 = probs.topk(3)

    st.subheader("Prediction")
    for prob, idx in zip(top3.values, top3.indices):
        st.progress(float(prob), text=f"{CLASS_NAMES[idx]}  —  {prob*100:.1f}%")
