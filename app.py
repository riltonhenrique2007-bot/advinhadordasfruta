import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Título do Site
st.set_page_config(page_title="Detector de Frutas", page_icon="🍎")
st.title("🍎 Detector de Frutas Inteligente")
st.write("Suba uma foto e minha IA vai dizer o que é!")

# Carrega a IA
@st.cache_resource
def load_model():
    # compile=False resolve o erro de versão do Keras
    return tf.keras.models.load_model('minha_ia.h5', compile=False)

model = load_model()
NOMES = ["LARANJA", "MAÇÃ", "MIRTILO", "PITAYA"]

# Botão de subir foto
foto = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "webp"])

if foto is not None:
    img = Image.open(foto)
    # use_container_width tira o aviso amarelo
    st.image(img, caption='Foto enviada', use_container_width=True)

    # Prepara para a IA
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predição
    preds = model.predict(img_array)[0] # O [0] garante que pegamos a lista de probabilidades
    idx = np.argmax(preds)

    st.success(f"Isso é uma: **{NOMES[idx]}**! (Confiança: {preds[idx] * 100:.2f}%)")
