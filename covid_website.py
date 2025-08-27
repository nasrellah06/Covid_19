import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 🔹 إخفاء تحذيرات TensorFlow الغير مهمة
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- عنوان التطبيق ---
st.title("🩺 Détection Covid à partir de Radiographies (Deep Learning)")

# --- تحميل الموديل المدرب ---
MODEL_PATH = "meilleur_model_covid_RMS.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle: {e}")
    st.stop()

# --- معرفة شكل الإدخال الذي يتوقعه الموديل ---
input_shape = model.input_shape  # ex: (None, 224, 224, 3)
_, H, W, C = input_shape
st.info(f"✅ Le modèle attend des images de taille: {H}x{W} avec {C} canaux")

# --- رفع صورة من المستخدم ---
uploaded_file = st.file_uploader("📤 Uploader une radiographie", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # اختيار RGB أو Grayscale حسب شكل الموديل
    if C == 3:
        image = Image.open(uploaded_file).convert("RGB").resize((H, W))
    elif C == 1:
        image = Image.open(uploaded_file).convert("L").resize((H, W))
    else:
        st.error("⚠️ Le modèle a un format d'entrée non supporté.")
        st.stop()

    # عرض الصورة
    st.image(image, caption="Image téléchargée", use_container_width=True)

    # --- Prétraitement ---
    img_array = np.array(image) / 255.0
    if C == 1:  # grayscale → ajouter une dimension canal
        img_array = np.expand_dims(img_array, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)  # (1,H,W,C)

    # --- Prédiction ---
    prediction = model.predict(img_array)

    # Si sortie = un seul neurone Sigmoid → binaire
    if prediction.shape[1] == 1:
        score = prediction[0][0]
        result = "Covid / Anormal" if score > 0.5 else "Normal"
        st.write(f"### Résultat : **{result}** (Score={score:.2f})")

    # Si sortie = softmax avec 2 classes
    elif prediction.shape[1] == 2:
        class_idx = np.argmax(prediction)
        classes = ["Normal", "Covid/Anormal"]
        result = classes[class_idx]
        st.write(f"### Résultat : **{result}** (Probabilités={prediction[0]})")

    else:
        st.error("⚠️ Format de sortie du modèle non reconnu.")
