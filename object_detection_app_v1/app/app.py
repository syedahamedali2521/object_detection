import streamlit as st
import numpy as np
import cv2
import tempfile
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Neon YOLO  Detector", page_icon="🧿", layout="wide")

# Load CSS
def load_css():
    css_path = Path(__file__).parent / "static" / "style.css"
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Header
st.markdown("""
<div class="header">
  <h1 class="title">NEON <span>OBJECT</span> DETECTOR</h1>
  <p class="sub">YOLOv8 - Image & Video - Futuristic Neon UI</p>
            <p>build by ❤️ Syed Ahamed Ali</p>
</div>
""", unsafe_allow_html=True)

# Load YOLO model
@st.cache_resource(show_spinner="Downloading YOLOv8s weights...")
def load_model():
    return YOLO("yolov8s.pt")   # ✅ Better accuracy than yolov8n

model = load_model()

# Tabs
tab_img, tab_vid, tab_about = st.tabs(["Image", "Video", "About"])

# =========================================================
# ✅ IMAGE TAB
# =========================================================
with tab_img:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col_left, col_right = st.columns([1,1])

    with col_left:
        img_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"])
        conf = st.slider("Confidence", 0.1, 0.9, 0.25, 0.05)
        iou = st.slider("NMS IoU", 0.1, 0.9, 0.45, 0.05)
        draw_labels = st.checkbox("Show labels", value=True)
        draw_conf = st.checkbox("Show confidence", value=True)
        neon = st.checkbox("Neon effect on preview", value=True)

    with col_right:
        if img_file is not None:
            img = Image.open(img_file).convert("RGB")
            st.image(img, caption="Uploaded", use_container_width=True)
if img_file is not None and st.button("Detect Objects", use_container_width=True):
    with st.spinner("Running YOLOv8..."):
        result = model.predict(img, conf=conf, iou=iou, verbose=False)

        # ✅ YOLO auto-draws labels + confidence
        plotted = result[0].plot()

        # ✅ Extract detected class names (OPTIONAL but useful)
        boxes = result[0].boxes
        names = model.model.names

        detected_list = []
        for box in boxes:
            cls = int(box.cls[0])
            label_name = names[cls]
            conf_score = float(box.conf[0])
            detected_list.append(f"{label_name} ({conf_score:.2f})")

    # ✅ Show list of detected objects
    if detected_list:
        st.markdown("### ✅ Detected Objects:")
        st.write(", ".join(detected_list))
    else:
        st.warning("No objects detected.")

    # ✅ Show detection result image (with neon option)
    if neon:
        st.markdown('<div class="neon-frame glow">', unsafe_allow_html=True)
        st.image(plotted, caption="Detections", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.image(plotted, caption="Detections", use_container_width=True)



        if neon:
            st.markdown('<div class="neon-frame glow">', unsafe_allow_html=True)
            st.image(plotted, caption="Detections", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.image(plotted, caption="Detections", use_container_width=True)

        # Download button
        out_path = Path(tempfile.gettempdir()) / f"detected_{img_file.name}"
        Image.fromarray(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)).save(out_path)

        with open(out_path, "rb") as f:
            st.download_button("Download result", f, file_name=out_path.name, mime="image/png")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# ✅ VIDEO TAB
# =========================================================
with tab_vid:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col_left, col_right = st.columns([1,1])

    with col_left:
        vid_file = st.file_uploader("Upload a video (mp4, mov, mkv, avi)", type=["mp4","mov","mkv","avi"])
        conf_v = st.slider("Confidence", 0.1, 0.9, 0.25, 0.05, key="conf_v")
        iou_v = st.slider("NMS IoU", 0.1, 0.9, 0.45, 0.05, key="iou_v")
        max_frames = st.number_input("Max frames to process (0 = all)", 0, 10000, 0, 1)
        show_preview = st.checkbox("Show preview frames", value=True)

    with col_right:
        if vid_file is not None:
            st.video(vid_file)

    if vid_file is not None and st.button("Detect in Video", use_container_width=True):
        import os

        # Save temp video
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid_file.name).suffix) as tfile:
            tfile.write(vid_file.read())
            tmp_input_path = tfile.name

        cap = cv2.VideoCapture(tmp_input_path)
        if not cap.isOpened():
            st.error("Could not read the video file.")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = Path(tempfile.gettempdir()) / ("detected_" + Path(tmp_input_path).stem + ".mp4")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            limit = total if max_frames == 0 else min(total, max_frames)
            prog = st.progress(0)
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    break

                res = model.predict(frame, conf=conf_v, iou=iou_v, verbose=False)
                plotted = res[0].plot()
                writer.write(plotted)

                # ✅ FIX: use_container_width instead of deprecated parameter
                if show_preview and (frame_idx % max(int(fps // 2), 1) == 0):
                    st.image(plotted, caption=f"Frame {frame_idx}", use_container_width=True)

                prog.progress(min(frame_idx / limit, 1.0))

            cap.release()
            writer.release()

            st.success("Done! Download your processed video below.")
            with open(out_path, "rb") as f:
                st.download_button("Download video", f, file_name=out_path.name, mime="video/mp4")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# ✅ ABOUT TAB
# =========================================================
with tab_about:
    st.markdown("""
<div class="card">
  <h3>About</h3>
  <p>This app uses YOLOv8s (Ultralytics) for object detection in images and videos.  
  The model downloads automatically on first use.</p>
  <ul>
    <li>Image & video detection</li>
    <li>Confidence & IoU control</li>
    <li>Futuristic neon theme</li>
  </ul>
</div>
""", unsafe_allow_html=True)

