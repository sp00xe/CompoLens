import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO

st.title("CompoLens: Saliency-Based Composition Assistant")

# Rule of Thirds Guide
def draw_rule_of_thirds(image_pil, color):
    width, height = image_pil.size
    draw = ImageDraw.Draw(image_pil)
    x1 = width / 3
    x2 = 2 * width / 3
    y1 = height / 3
    y2 = 2 * height / 3
    draw.line([(x1, 0), (x1, height)], fill=color, width=2)
    draw.line([(x2, 0), (x2, height)], fill=color, width=2)
    draw.line([(0, y1), (width, y1)], fill=color, width=2)
    draw.line([(0, y2), (width, y2)], fill=color, width=2)
    return image_pil

def draw_center_lines(image_pil, color):
    width, height = image_pil.size
    draw = ImageDraw.Draw(image_pil)
    cx = width / 2
    cy = height / 2
    draw.line([(cx, 0), (cx, height)], fill=color, width=2)
    draw.line([(0, cy), (width, cy)], fill=color, width=2)
    return image_pil

def draw_diagonals(image_pil, color):
    width, height = image_pil.size
    draw = ImageDraw.Draw(image_pil)
    draw.line([(0, 0), (width, height)], fill=color, width=2)
    draw.line([(width, 0), (0, height)], fill=color, width=2)
    return image_pil

def draw_golden_ratio(image_pil, color):
    width, height = image_pil.size
    draw = ImageDraw.Draw(image_pil)
    phi = 0.618
    x1 = width * phi
    y1 = height * phi
    draw.line([(x1, 0), (x1, height)], fill=color, width=2)
    draw.line([(0, y1), (width, y1)], fill=color, width=2)
    return image_pil

# ---------- UI Controls ----------

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="image_uploader")
guide_type = st.selectbox("Choose Compositional Guide", ["None", "Rule of Thirds", "Center Lines", "Diagonals", "Golden Ratio"])
overlay_color = st.color_picker("Overlay Color", "#FF0000")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Apply selected overlay
    if guide_type == "Rule of Thirds":
        st.subheader("Image with Rule of Thirds Overlay")
        image_to_display = draw_rule_of_thirds(image.copy(), overlay_color)
    elif guide_type == "Center Lines":
        st.subheader("Image with Center Lines Overlay")
        image_to_display = draw_center_lines(image.copy(), overlay_color)
    elif guide_type == "Diagonals":
        st.subheader("Image with Diagonals Overlay")
        image_to_display = draw_diagonals(image.copy(), overlay_color)
    elif guide_type == "Golden Ratio":
        st.subheader("Image with Golden Ratio Overlay")
        image_to_display = draw_golden_ratio(image.copy(), overlay_color)
    else:
        st.subheader("Original Image")
        image_to_display = image

    st.image(image_to_display, use_container_width=True)


    # Initialize OpenCV's static saliency model
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image_bgr)

    if success:
        saliencyMap = (saliencyMap * 255).astype("uint8")
        saliencyMap_colored = cv2.applyColorMap(saliencyMap, cv2.COLORMAP_JET)
        saliencyMap_rgb = cv2.cvtColor(saliencyMap_colored, cv2.COLOR_BGR2RGB)

        st.image(saliencyMap_rgb, caption="Saliency Map", use_container_width=True)
    else:
        st.warning("Failed to generate saliency map. Try a different image.")
