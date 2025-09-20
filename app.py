import streamlit as st
from ultralytics import YOLO
import numpy as np, cv2, json, os
from PIL import Image

st.set_page_config(page_title="Detector YOLO", page_icon="🔎", layout="centered")
st.title("🔎 Demo YOLO: Subí una imagen y evaluá la detección")

st.markdown(
    """
**Modo actual:** `yolov11m.pt` (COCO, se descarga automáticamente).  
Este modelo **no** detecta enfermedades de cacao; sirve como demo.  
Cuando tengas tu modelo entrenado, cambia la ruta en `DEFAULT_MODEL` más abajo.
"""
)

# ---------- CONFIG ----------
# Por ahora usamos el modelo general de Ultralytics (se descarga solo):
DEFAULT_MODEL = "yolov11m.pt"
# Cuando tengas tu modelo:
# DEFAULT_MODEL = "models/best.onnx"  # recomendado en CPU
# DEFAULT_MODEL = "models/best.pt"    # opción PyTorch

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

model = load_model(DEFAULT_MODEL)
NAMES = getattr(model, "names", None) if hasattr(model, "names") else None

# -------- Panel de parámetros ----------
conf  = st.slider("Confianza (conf)", 0.05, 0.90, 0.25, 0.05)
iou   = st.slider("IoU", 0.30, 0.90, 0.60, 0.05)
imgsz = st.slider("Tamaño de entrada (imgsz)", 320, 960, 640, 64)

file = st.file_uploader("Sube una imagen", type=["jpg","jpeg","png"])

run = False
if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Vista previa", use_column_width=True)
    # Botón para evaluar
    run = st.button("🔍 Evaluar imagen")

def infer(image: Image.Image):
    img = np.array(image.convert("RGB"))
    res = model.predict(source=img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

    # Visualizar
    plotted = res.plot()             # BGR
    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

    # Conteo por clase
    counts = {}
    for b in res.boxes:
        cls = int(b.cls.item())
        name = NAMES.get(cls, str(cls)) if NAMES else str(cls)
        counts[name] = counts.get(name, 0) + 1

    total = sum(counts.values())
    return plotted, {**counts, "total": total}

if run:
    vis, counts = infer(image)
    st.image(vis, caption="Detecciones", use_column_width=True)
    st.subheader("Conteo por clase")
    st.json(counts)

st.info(
    "🔁 **Cambiar a tu modelo:** sube `models/best.onnx` (o `best.pt`) al repo "
    "y cambia `DEFAULT_MODEL` en el código."
)
