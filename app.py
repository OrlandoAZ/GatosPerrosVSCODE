"""
Clasificador de Perros y Gatos — Streamlit
Pipeline exacto del notebook Despliegue-4000-15-06-2026.ipynb
"""

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input

# ── Ruta del modelo — ajusta si cambias de equipo ───────────────────────────
RUTA_MODELO = r"E:\OAZ\0D. PYTHON\VS Transfer Learning\Codigo optimizado 2026-4000VS\head_final4000VS.keras"

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Perros y Gatos",
    page_icon="🐾",
    layout="centered",
)

# ── Encabezado ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center; color:#2c3e50;'>🐾 Clasificador de Perros y Gatos</h1>
    <p style='text-align:center; color:#7f8c8d; font-size:17px;'>
        Transfer Learning con <strong>ResNet152</strong>
    </p>
    <hr style='border:1px solid #ecf0f1;'>
    """,
    unsafe_allow_html=True,
)

# ── Carga de modelos (una sola vez por sesión) ───────────────────────────────
@st.cache_resource(show_spinner="Cargando modelos…")
def cargar_modelos():
    base = ResNet152(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base.trainable = False
    head = load_model(RUTA_MODELO)
    return base, head

base_model, head = cargar_modelos()

# ── Subida de imagen ─────────────────────────────────────────────────────────
st.markdown("### 📤 Sube una imagen")
archivo = st.file_uploader(
    "Selecciona una foto de gato o perro (.jpg / .jpeg / .png)",
    type=["jpg", "jpeg", "png"],
)

# ── Inferencia ───────────────────────────────────────────────────────────────
if archivo is not None:

    # Leer imagen igual que cv2.imread → BGR→RGB
    file_bytes = np.frombuffer(archivo.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Preprocesar → features → predecir  (mismo pipeline del notebook)
    img_pre = preprocess_input(cv2.resize(img_rgb, (224, 224)).astype('float32'))
    prob    = head.predict(
                  base_model.predict(np.expand_dims(img_pre, 0), verbose=0),
                  verbose=0
              )[0][0]

    clase     = 'Perro' if prob > 0.5 else 'Gato'
    confianza = prob if prob > 0.5 else 1 - prob
    nombre    = archivo.name
    color     = "#e67e22" if prob > 0.5 else "#2980b9"
    emoji     = "🐶" if prob > 0.5 else "🐱"

    # ── Resultado ────────────────────────────────────────────────────────────
    st.markdown("<hr style='border:1px solid #ecf0f1;'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Resultado")

    col_img, col_info = st.columns([1, 1], gap="large")

    with col_img:
        st.image(img_rgb, caption=nombre, use_container_width=True)

    with col_info:
        st.markdown(
            f"""
            <div style='
                background:#f8f9fa;
                border-left:6px solid {color};
                border-radius:8px;
                padding:24px 20px;
                margin-top:8px;
            '>
                <p style='margin:0 0 6px 0;color:#888;font-size:13px;'>📁 ARCHIVO</p>
                <p style='margin:0 0 22px 0;font-size:15px;word-break:break-all;color:#2c3e50;'>
                    {nombre}
                </p>
                <p style='margin:0 0 6px 0;color:#888;font-size:13px;'>🏷️ PREDICCIÓN</p>
                <p style='margin:0 0 22px 0;font-size:32px;font-weight:bold;color:{color};'>
                    {emoji} {clase}
                </p>
                <p style='margin:0 0 6px 0;color:#888;font-size:13px;'>📊 PROBABILIDAD</p>
                <p style='margin:0;font-size:32px;font-weight:bold;color:{color};'>
                    {float(confianza) * 100:.1f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(float(confianza), text=f"Confianza: {float(confianza)*100:.1f}%")

# instalar:   python.exe -m pip install streamlit
# streamlit run app.py

#------> Sale correcto!!!!

#python.exe -m pip freeze > requirements.txt