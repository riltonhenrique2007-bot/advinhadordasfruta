import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Título do Site
st.title("🍎 Detector de Frutas Inteligente")
st.write("Suba uma foto e minha IA vai dizer o que é!")


# Carrega a IA
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('minha_ia.h5')


model = load_model()
NOMES = ["LARANJA", "MAÇÃ", "MIRTILO", "PITAYA"]

# Botão de subir foto
foto = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "webp"])

if foto is not None:
    img = Image.open(foto)
    st.image(img, caption='Foto enviada', use_column_width=True)

    # Prepara para a IA
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predição
    preds = model.predict(img_array)
    idx = np.argmax(preds)

    st.success(f"Isso é uma: **{NOMES[idx]}**! (Confiança: {preds[0][idx] * 100:.2f}%)")
