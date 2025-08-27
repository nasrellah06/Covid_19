import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

st.title("Détection Covid (Keras Model)")

# --- Charger le modèle Keras ---
model = tf.keras.models.load_model("meilleur_model_covid_RMS.keras")

# --- Uploader image ---
uploaded_file = st.file_uploader("Uploader une radiographie", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((224,224))
    st.image(image, caption="Image téléchargée", use_column_width=True)

    # Prétraitement
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction
    prediction = model.predict(img_array)
    result = "Covid" if prediction[0][0] > 0.5 else "Normal"

    st.write(f"Résultat : **{result}** (Score={prediction[0][0]:.2f})")
