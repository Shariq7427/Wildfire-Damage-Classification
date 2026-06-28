import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Wildfire Damage Classification",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# Custom CSS
# ==========================
# ===========================================
# HERO SECTION
# ===========================================

st.markdown("""
<style>

.hero{

background: linear-gradient(90deg,#7f1d1d,#dc2626,#f97316);

padding:40px;

border-radius:20px;

color:white;

text-align:center;

margin-bottom:30px;

box-shadow:0px 10px 25px rgba(0,0,0,.2);

}

.hero h1{

font-size:48px;

margin-bottom:10px;

}

.hero p{

font-size:20px;

color:white;

}

.metric-card{

background:white;

padding:18px;

border-radius:15px;

text-align:center;

box-shadow:0 4px 12px rgba(0,0,0,.15);

}

.prediction-card{

background:#fff8e6;

padding:25px;

border-radius:15px;

border-left:8px solid #ff6b00;

box-shadow:0 4px 10px rgba(0,0,0,.1);

}

</style>
""", unsafe_allow_html=True)

st.markdown("""

<div class="hero">

<h1>🔥 Wildfire Damage Classification AI</h1>

<p>
Deep Learning Based Structural Damage Assessment
using Convolutional Neural Networks
</p>

</div>

""", unsafe_allow_html=True)

col1,col2,col3,col4=st.columns(4)

with col1:
    st.metric("Dataset","6 Classes")

with col2:
    st.metric("Model","CNN")

with col3:
    st.metric("Image Size","224×224")

with col4:
    st.metric("Framework","TensorFlow")

# ==========================
# Sidebar
# ==========================
st.sidebar.image("https://img.icons8.com/color/96/fire-element.png", use_container_width=True)
width=80
st.sidebar.title("Wildfire AI")

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded")

st.sidebar.info("""
CNN Architecture

Image Size: 224×224

Framework: TensorFlow

Classes: 6
""")

st.sidebar.markdown("---")

st.sidebar.write("Developed by")

st.sidebar.write("**Shariq Ali**")

st.sidebar.markdown(
"[GitHub Repository](https://github.com/Shariq7427/Wildfire-Damage-Classification)"
)

# ==========================
# Title
# ==========================
st.markdown(
    "<p class='big-title'>🔥 Wildfire Damage Classification using CNN</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub-title'>AI-powered structural damage assessment from wildfire images.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("notebooks/models/cnn_model.h5")

model = load_model()

# ==========================
# Class Labels
# ==========================
class_names = [
    "affected",
    "destroyed",
    "inaccessible",
    "major",
    "minor",
    "no_damage"
]

# ==========================
# Upload Image
# ==========================

uploaded_file = st.file_uploader(
    "📤 Upload a Wildfire Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, use_container_width=True)

    with col2:
        if st.button("🔥 Predict Damage"):

            img = image.resize((224,224))
            img = np.array(img).astype("float32")/255.0
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)
            probabilities = prediction[0]

            predicted_index = np.argmax(probabilities)
            predicted_class = class_names[predicted_index]
            confidence = probabilities[predicted_index] * 100

            st.success(f"{predicted_class} ({confidence:.2f}%)")

            # now SAFE to build chart
            fig = go.Figure(...)
            st.plotly_chart(fig)
# ==========================
# Prediction Probabilities
# ==========================

st.subheader("📊 Prediction Probabilities")

fig = go.Figure(
    go.Bar(
        x=probabilities * 100,
        y=class_names,
        orientation="h",
        text=[f"{p*100:.2f}%" for p in probabilities],
        textposition="outside",
        marker=dict(
            color=[
                "#e63946",
                "#f77f00",
                "#fcbf49",
                "#90be6d",
                "#43aa8b",
                "#577590"
            ]
        )
    )
)

fig.update_layout(
    height=350,
    xaxis_title="Confidence (%)",
    yaxis_title="Damage Class",
    template="plotly_white",
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig, use_container_width=True)
severity = {
    "destroyed": "🔴 Critical Damage",
    "major": "🟠 Major Damage",
    "minor": "🟡 Minor Damage",
    "affected": "🟢 Affected",
    "no_damage": "✅ No Damage",
    "inaccessible": "⚪ Unable to Assess"
}

st.info(f"### {severity[predicted_class]}")
st.subheader("Confidence")

st.progress(float(confidence / 100))

st.write(f"Model Confidence: **{confidence:.2f}%**")
st.markdown("---")

with st.expander("📖 About this Project"):

    st.write("""
This application automatically classifies wildfire
damage using a Convolutional Neural Network trained
on aerial images.

### Features

- Upload wildfire images

- CNN prediction

- Confidence score

- Six damage categories

### Technologies

- TensorFlow

- Streamlit

- NumPy

- Pillow

- Python
""")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    st.markdown("## 🧠 CNN Model")

    st.write("""
- Convolutional Neural Network
- TensorFlow / Keras
- Input Size: 224×224
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
""")

with col2:

    st.markdown("## 📂 Dataset")

    st.write("""
Structure Wildfire Damage Classification Dataset

Classes:

- affected
- destroyed
- inaccessible
- major
- minor
- no_damage
""")
st.markdown("---")

st.markdown(
"""
<div style='text-align:center;color:gray;'>

### 🔥 Wildfire Damage Classification AI

Developed by <b>Shariq Ali</b>

TensorFlow • Streamlit • Deep Learning

</div>
""",
unsafe_allow_html=True
)
