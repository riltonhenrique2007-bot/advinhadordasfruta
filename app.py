import os
os.environ['TF_USE_LEGACY_KERAS'] = '1' # Ativa o modo clássico

import streamlit as st
import tensorflow as tf
import tf_keras as keras # Importamos a versão de compatibilidade
from PIL import Image
import numpy as np

# Título
st.set_page_config(page_title="Detector de Frutas", page_icon="🍎")
st.title("🍎 Detector de Frutas Inteligente")

# Carrega a IA usando o keras (não o tf.keras)
@st.cache_resource
def load_model():
    return keras.models.load_model('minha_ia.h5', compile=False)

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
