import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


# ==========================
# Load Model & Labels
# ==========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/cataract_model_best.keras")
    return model


model = load_model()

with open("models/labels.json") as f:
    idx_to_class = json.load(f)

# Pastikan list sesuai urutan indeks
classes = [
    (
        idx_to_class[str(i)]
        if isinstance(list(idx_to_class.keys())[0], str)
        else idx_to_class[i]
    )
    for i in range(len(idx_to_class))
]


# ==========================
# Preprocessing
# ==========================
def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.array(img).astype(np.float32)
    x = preprocess_input(x)  # ✅ sama dengan training
    return np.expand_dims(x, axis=0)


# ==========================
# Streamlit UI
# ==========================
st.title("🩺 Deteksi Katarak")

uploaded_file = st.file_uploader("Upload gambar mata", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar diupload", use_column_width=True)

    if st.button("Prediksi"):
        X = preprocess_image(image)
        prob = model.predict(X, verbose=0)[0]  # softmax output
        idx = int(np.argmax(prob))
        confidence = float(np.max(prob))
        label = classes[idx]

        # ==========================
        # Validasi: apakah ini gambar mata?
        # ==========================
        if confidence < 0.6:  # threshold 60%
            st.error("❌ Gambar tidak dikenali sebagai gambar mata yang valid.")
        else:
            st.markdown(f"### Hasil Prediksi: **{label}** ({confidence*100:.2f}%)")

            st.write(
                {classes[0]: f"{prob[0]*100:.2f}%", classes[1]: f"{prob[1]*100:.2f}%"}
            )
