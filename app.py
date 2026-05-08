import os
# FORÇA O SERVIDOR A USAR O KERAS ANTIGO (ESSENCIAL!)
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Configuração da aba do navegador
st.set_page_config(page_title="Detector de Frutas", page_icon="🍎")

st.title("🍎 Detector de Frutas Inteligente")
st.write("Suba uma foto e minha IA vai dizer o que é!")

# Função para carregar o modelo de um jeito que o Linux aceite
@st.cache_resource
def load_my_model():
    # compile=False evita erros de versão entre Windows e Linux
    model = tf.keras.models.load_model("minha_ia.h5", compile=False)
    return model

model = load_my_model()
NOMES = ["LARANJA", "MAÇÃ", "MIRTILO", "PITAYA"]

# Botão de subir foto
foto = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "webp"])

if foto is not None:
    # Mostra a imagem na tela
    image = Image.open(foto).convert("RGB")
    st.image(image, caption='Foto enviada', use_container_width=True)

    # PREPARAÇÃO DA IMAGEM (PADRÃO TEACHABLE MACHINE)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    # Normalização dos pixels
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Cria o "pacote" para a IA processar
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Faz a predição
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = NOMES[index]
    confidence_score = prediction[0][index]

    # Resultado final
    st.success(f"Isso é uma: **{class_name}**! (Confiança: {confidence_score * 100:.2f}%)")
