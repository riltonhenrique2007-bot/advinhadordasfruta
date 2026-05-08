import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Configuração da página
st.set_page_config(page_title="Detector de Frutas", page_icon="🍎")
st.title("🍎 Detector de Frutas Inteligente")

# Função para carregar o modelo de um jeito mais "robusto"
@st.cache_resource
def load_my_model():
    # Carrega sem compilar para evitar erros de versão no servidor Linux
    model = tf.keras.models.load_model("minha_ia.h5", compile=False)
    return model

model = load_my_model()
NOMES = ["LARANJA", "MAÇÃ", "MIRTILO", "PITAYA"]

foto = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "webp"])

if foto is not None:
    image = Image.open(foto).convert("RGB")
    st.image(image, caption='Foto enviada', use_container_width=True)

    # O Teachable Machine exige esse pré-processamento exato:
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predição
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = NOMES[index]
    confidence_score = prediction[0][index]

    st.success(f"Isso é uma: **{class_name}**! (Confiança: {confidence_score * 100:.2f}%)")
