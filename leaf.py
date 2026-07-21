import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from huggingface_hub import InferenceClient

PROJECT_DIR = r"C:\Users\pc\Desktop\finalproject"
TRAIN_DIR = os.path.join(PROJECT_DIR, 'train')
MODEL_PATH = os.path.join(PROJECT_DIR, 'best_mobilenetv2_model.pth')

TOKEN_INPUT = "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"  

NUM_CLASSES = 38 

if os.path.exists(TRAIN_DIR):
    folder_names = sorted(os.listdir(TRAIN_DIR))
    class_mapping = {i: folder_name for i, folder_name in enumerate(folder_names)}
else:
    class_mapping = {i: f"Unknown_Disease_{i}" for i in range(NUM_CLASSES)}
    class_mapping.update({
        0: "Tomato___Bacterial_spot", 
        1: "Potato___Early_blight", 
        2: "Corn___Common_rust", 
        3: "Tomato___Healthy",
        4: "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    })

@st.cache_resource
def load_and_prepare_model():
    """Builds MobileNetV2 architecture and maps saved checkpoint weights securely."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None) 
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(model.last_channel, NUM_CLASSES)
    )
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
    else:
        st.sidebar.warning("⚠️ Weights file not found. Running model on uninitialized layers.")
        
    model.to(device)
    model.eval()
    return model, device

model_instance, device = load_and_prepare_model()

def process_and_predict(image, model, exec_device):
    """Preprocesses a PIL image and extracts deep neural predictions."""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(image).unsqueeze(0).to(exec_device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probabilities, 1)
        
    predicted_disease = class_mapping.get(pred_idx.item(), f"Unknown_Class_{pred_idx.item()}")
    confidence_score = confidence.item() * 100
    
    return predicted_disease, confidence_score

def get_llm_recommendation(disease_name, hf_token=None):
    """Fetches structured plant pathology guidance via Hugging Face Chat Completion API."""
    if "healthy" in disease_name.lower():
        return "🌱 **Your crop is completely healthy!** Continue routine watering of crops and monitor the fields regularly."
        
    try:
        client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=hf_token)
        
        clean_name = disease_name.replace("___", " ").replace("_", " ")
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert agricultural scientist and plant pathologist. "
                    "Analyze the specific leaf disease requested. Pay meticulous attention to whether "
                    "it is a viral, bacterial, fungal, or insect-driven infection. Provide context-accurate "
                    "treatments—for example, if it is viral, explicitly state there are no direct chemical cures "
                    "and emphasize vector insect control. Give direct, farmer-friendly advice. Do not write "
                    "introductory conversational text. Start immediately with the requested headers."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Disease detected: {clean_name}. Provide a structured diagnostic report using these exact Markdown headers:\n\n"
                    f"### 1. Symptoms:\n[Write a brief 2-sentence description tailored explicitly to {clean_name}]\n\n"
                    f"### 2. Causes:\n[Explain the specific pathogen vectors, transmission modes, or environmental triggers for {clean_name}]\n\n"
                    f"### 3. Treatment Recommendations:\n[Provide 2 to 3 actionable, direct steps for management, tailored specifically to whether it's fungal, viral, or bacterial]\n\n"
                    f"### 4. Prevention Tips:\n[Provide 2 preventive strategies to secure future crop planting cycles]"
                )
            }
        ]
        
        response = client.chat_completion(messages=messages, max_tokens=500, temperature=0.2)
        
        return response.choices[0].message.content

    except Exception as e:
        return (
            "### 1. General Symptoms:\n"
            "Leaves change shape. Strange spots appear. Leaves looks unhealthy. Visual signs depend on the disease.\n\n"
            "### 2. General Causes:\n"
            "Harmful germs grow. Soil stays too wet. Plants suffer from stress. Moving bugs spread sickness.\n\n"
            "### 3. General Treatments:\n"
            "* Cut off sick branches. Remove dying crops immediately. Use correct spray options. Apply insecticide for bugs. Use fungicide for mold.\n"
            "### 4. General Prevention:\n"
            "Buy certified healthy plants. Avoid cheap, risky seeds. Wash tools after use. Clean gear between fields.\n"
        )

st.set_page_config(page_title="Leaf disease detection model", page_icon="🌿", layout="wide")
st.title("🌿 Welcome to my Final Project.")
st.markdown("In this project, I have built a leaf disease detection model using the CNN deep learning model architecture.")

with st.sidebar:
    selected = option_menu("Main Menu", ["Home", "Leaf Disease Detection"], 
        icons=['house', 'search'], menu_icon="cast", default_index=1)

use_token = TOKEN_INPUT if TOKEN_INPUT and TOKEN_INPUT != "hf_yourActualTokenStringHere" else None

if selected == "Home":
    st.header('Leaf Disease Detection Model')
    st.write("Plant diseases significantly reduce agricultural productivity and crop quality worldwide. Farmers often struggle to identify diseases accurately, especially during early stages. To help solve this problem, I have built an AI-Powered Plant Disease Detection and Treatment Recommendation model using Deep Learning and Hugging Face Generative AI.")
    
    if os.path.exists("leaf.jpg"):
        st.image("leaf.jpg", caption="Jessica Sriramula")
    else:
        st.info("Place 'leaf.jpg' in your project directory to display your home screen banner.")

elif selected == "Leaf Disease Detection":
    uploaded_file = st.file_uploader("Please upload an image of your leaf plant by clicking on the upload button here:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Leaf image", use_container_width=True)
            
        with col2:
            st.subheader("📊 Leaf image diagnosis")
            with st.spinner("Analyzing leaf image..."):
                predicted_disease, confidence_score = process_and_predict(image, model_instance, device)
                
            clean_display_name = predicted_disease.replace('___', ' ').replace('_', ' ')
            st.success(f"Name: **{clean_display_name}**")
            st.metric(label="Model Confidence Score", value=f"{confidence_score:.2f}%")
            st.progress(confidence_score / 100.0)
            
        st.markdown("---")
        st.subheader("📋 Notes")
        with st.spinner("Loading, please wait..."):
            treatment_plan = get_llm_recommendation(predicted_disease, hf_token=use_token)
            st.markdown(treatment_plan)