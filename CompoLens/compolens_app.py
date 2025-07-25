import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from streamlit_drawable_canvas import st_canvas
from io import BytesIO
from math import hypot

st.title("CompoLens: Saliency-Based Composition Assistant")

# ---------- Settings ----------
MAX_CANVAS_WIDTH = 1000  # pixels

# ---------- Drawing Helpers ----------
def draw_overlay(image_pil, guide_type, color):
    width, height = image_pil.size
    draw = ImageDraw.Draw(image_pil)

    if guide_type == "Rule of Thirds":
        draw.line([(width / 3, 0), (width / 3, height)], fill=color, width=2)
        draw.line([(2 * width / 3, 0), (2 * width / 3, height)], fill=color, width=2)
        draw.line([(0, height / 3), (width, height / 3)], fill=color, width=2)
        draw.line([(0, 2 * height / 3), (width, 2 * height / 3)], fill=color, width=2)

    elif guide_type == "Center Lines":
        draw.line([(width / 2, 0), (width / 2, height)], fill=color, width=2)
        draw.line([(0, height / 2), (width, height / 2)], fill=color, width=2)

    elif guide_type == "Diagonals":
        draw.line([(0, 0), (width, height)], fill=color, width=2)
        draw.line([(width, 0), (0, height)], fill=color, width=2)

    elif guide_type == "Golden Ratio":
        phi = 0.618
        draw.line([(width * phi, 0), (width * phi, height)], fill=color, width=2)
        draw.line([(0, height * phi), (width, height * phi)], fill=color, width=2)

    return image_pil

def draw_focal_point(image_pil, x, y):
    draw = ImageDraw.Draw(image_pil)
    radius = 10
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="yellow", outline="black", width=2)
    return image_pil

# ---------- UI Controls ----------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
guide_type = st.selectbox("Choose Compositional Guide", ["None", "Rule of Thirds", "Center Lines", "Diagonals", "Golden Ratio"])
overlay_color = st.color_picker("Overlay Color", "#FF0000")
saliency_opacity = st.slider("Saliency Overlay Opacity", 0.0, 1.0, 0.5, 0.05)
show_saliency = st.checkbox("Show Saliency Map", value=True)
show_overlay = st.checkbox("Show Compositional Overlay", value=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Scale image if too large
    if image.width > MAX_CANVAS_WIDTH:
        scale = MAX_CANVAS_WIDTH / image.width
        image = image.resize((int(image.width * scale), int(image.height * scale)))

    width, height = image.size
    base_image = image.convert("RGBA")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.subheader("Click to Set Your Focal Point")

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=1,
        background_image=image,
        update_streamlit=True,
        height=height,
        width=width,
        drawing_mode="point",
        key="canvas"
    )

    final_image = base_image

    

    # ---------- Saliency Layer ----------
    if show_saliency:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliencyMap = saliency.computeSaliency(image_bgr)
        if success:
            saliencyMap = (saliencyMap * 255).astype("uint8")
            saliencyMap_colored = cv2.applyColorMap(saliencyMap, cv2.COLORMAP_JET)
            saliencyMap_rgb = cv2.cvtColor(saliencyMap_colored, cv2.COLOR_BGR2RGB)
            saliency_pil = Image.fromarray(saliencyMap_rgb).convert("RGBA")
            saliency_pil = ImageEnhance.Brightness(saliency_pil).enhance(0.9)
            saliency_layer = saliency_pil.resize((width, height)).convert("RGBA")
            final_image = Image.blend(final_image, saliency_layer, alpha=saliency_opacity)
        else:
            st.warning("Failed to generate saliency map.")


    # ---------- Overlay Layer ----------
    if show_overlay and guide_type != "None":
        final_image = draw_overlay(final_image, guide_type, overlay_color)

    # ---------- Focal Point ----------
    if canvas_result.json_data and "objects" in canvas_result.json_data:
        objects = canvas_result.json_data["objects"]
        circles = [obj for obj in objects if obj["type"] == "circle"]
        if circles:
            last_circle = circles[-1]
            x, y = int(last_circle["left"]), int(last_circle["top"])
            final_image = draw_focal_point(final_image, x, y)
            st.success(f"Focal Point Set at: ({x}, {y})")

            # Rule of Thirds Guidance
            thirds_x = [image.width / 3, 2 * image.width / 3]
            thirds_y = [image.height / 3, 2 * image.height / 3]
            target_points = [(tx, ty) for tx in thirds_x for ty in thirds_y]

            distances = [hypot(x - tx, y - ty) for (tx, ty) in target_points]
            min_distance = min(distances)
            closest_point = target_points[distances.index(min_distance)]

            threshold = 50  # pixels
            cx, cy = closest_point

            if min_distance < threshold:
                    st.markdown("Great framing = your focal point is close to a rule of thirds intersection!")
            else:
                    direction = []
                    if y < cy: direction.append("down")
                    elif y > cy: direction.append("up")
                    if x < cx: direction.append("right")
                    elif x > cx: direction.append("left")

                    if direction:
                        direction_text = " and ".join(direction)
                        st.markdown(f" Try moving the focal point slightly {direction_text} to improve composition.")

            #Saliency alignment check
            if show_saliency and 'saliencyMap' in locals() and success:
                    saliency_at_point = saliencyMap[min(y, saliencyMap.shape[0]-1), min(x, saliencyMap.shape[1]-1)]
                    if saliency_at_point > 0.5:
                        st.markdown(f"Your focal point aligns with a high-saliency area (value: {saliency_at_point:.2f})")
                    else:
                        st.markdown(f"Your focal point is in a low-attention region (value: {saliency_at_point:.2f})")
        if st.button("Clear Focal Point"):
            st.experimental_rerun()
    st.subheader("Final Image with Layers and Focal Point")
    st.image(final_image)

    buf = BytesIO()
    final_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Final Image", data=byte_im, file_name="scaled_canvas_output.png", mime="image/png")