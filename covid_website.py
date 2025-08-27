import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# --- Charger le modèle ---
model = tf.keras.models.load_model("meilleur_model_covid_RMS.keras")

# Classes (à adapter selon ton dataset)
CLASSES = ["Normal", "Covid", "Pneumonia"]

st.title("🩺 Détection Automatique : Covid / Pneumonie / Normal")
st.write("Uploadez une radiographie (X-ray ou CT) et le modèle prédira la classe.")

# --- Upload Image ---
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128, 128))  # adapter à ton modèle
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.image(img, caption="Image chargée", use_container_width=True)

    # --- Prédiction ---
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]

    # --- Résultat ---
    st.write(f"### ✅ Résultat : **{CLASSES[class_idx]}** (Confiance = {confidence:.2f})")

    # Afficher toutes les probabilités
    st.subheader("📊 Probabilités par classe :")
    for i, prob in enumerate(prediction[0]):
        st.write(f"- {CLASSES[i]} : {prob:.2f}")
