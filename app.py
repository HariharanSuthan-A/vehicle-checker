import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import io
import time
import base64
from datetime import datetime

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Checker",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #080c18 0%, #0f172a 60%, #080c18 100%);
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: rgba(8, 12, 30, 0.97);
    border-right: 1px solid rgba(99,102,241,0.2);
}

/* Hero */
.hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.18) 0%, rgba(168,85,247,0.18) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 1.8rem 2.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.hero-icon { font-size: 3.5rem; }
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub { color: #64748b; font-size: 0.9rem; margin: 0.2rem 0 0 0; }

/* Cards */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

/* Section heading */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #c7d2fe;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-bike { background:rgba(251,146,60,0.15); border:1px solid rgba(251,146,60,0.45); color:#fb923c; }
.badge-car  { background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.45); color:#818cf8; }

/* Log table */
.log-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.log-table th {
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    font-weight: 600;
    padding: 0.7rem 1rem;
    text-align: left;
    border-bottom: 1px solid rgba(99,102,241,0.2);
}
.log-table td {
    padding: 0.6rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    color: #cbd5e1;
    vertical-align: middle;
}
.log-table tr:hover td { background: rgba(99,102,241,0.06); }
.log-table img {
    border-radius: 6px;
    border: 1px solid rgba(99,102,241,0.3);
    max-height: 52px;
    width: auto;
}
.conf-pill {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
}
.conf-high { background:rgba(52,211,153,0.15); color:#34d399; border:1px solid rgba(52,211,153,0.3); }
.conf-med  { background:rgba(251,191,36,0.15); color:#fbbf24; border:1px solid rgba(251,191,36,0.3); }
.conf-low  { background:rgba(248,113,113,0.15); color:#f87171; border:1px solid rgba(248,113,113,0.3); }

/* Button */
.stButton > button {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 10px;
    color: #c7d2fe;
    font-weight: 600;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    background: rgba(99,102,241,0.28);
    border-color: rgba(99,102,241,0.75);
    color: #fff;
    transform: translateY(-1px);
    box-shadow: 0 4px 18px rgba(99,102,241,0.35);
}

hr { border-color: rgba(255,255,255,0.07) !important; }

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}

.no-log {
    text-align: center;
    color: #475569;
    padding: 2.5rem 1rem;
    font-size: 0.9rem;
    border: 1px dashed rgba(99,102,241,0.2);
    border-radius: 12px;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "🏍️  Bike / Motorcycle": {
        "path": os.path.join(BASE_DIR, "bikemodel.pt"),
        "badge": '<span class="badge badge-bike">🏍️ Bike</span>',
        "color": (251, 146, 60),
        "description": "License plates on bikes & motorcycles",
        "short": "Bike",
    },
    "🚗  Car / Truck": {
        "path": os.path.join(BASE_DIR, "carmodel.pt"),
        "badge": '<span class="badge badge-car">🚗 Car/Truck</span>',
        "color": (129, 140, 248),
        "description": "License plates on cars & trucks",
        "short": "Car/Truck",
    },
}

SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

BIKE_SAMPLES = [f for f in [
    "bike.jpeg","bike3.jpg","bike5.jpg","bike7.jpg",
    
] if os.path.exists(os.path.join(SAMPLES_DIR, f))]

CAR_SAMPLES = [f for f in [
    "car3.jpg","car5.jpg","car6.jpg",
] if os.path.exists(os.path.join(SAMPLES_DIR, f))]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str) -> YOLO:
    return YOLO(path)

def img_to_b64(img: Image.Image, size=(120, 60)) -> str:
    """Return a base-64 PNG thumbnail for inline HTML."""
    thumb = img.copy()
    thumb.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def conf_pill(conf: float) -> str:
    pct = f"{conf:.1%}"
    if conf >= 0.70:
        return f'<span class="conf-pill conf-high">{pct}</span>'
    elif conf >= 0.45:
        return f'<span class="conf-pill conf-med">{pct}</span>'
    else:
        return f'<span class="conf-pill conf-low">{pct}</span>'

def run_inference(model: YOLO, image: Image.Image, conf_thresh: float,
                  iou_thresh: float, box_color, model_short: str, source_name: str):
    """
    Run YOLO inference.
    Returns:
        annotated_img  – full image with drawn boxes
        log_entries    – list[dict] ready to append to st.session_state.log
    """
    rgb = image.convert("RGB")
    arr = np.array(rgb)
    results = model.predict(source=arr, conf=conf_thresh, iou=iou_thresh, verbose=False)
    result = results[0]

    annotated = rgb.copy()
    draw = ImageDraw.Draw(annotated)
    entries = []

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    r, g, b = box_color

    if result.boxes is not None and len(result.boxes) > 0:
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf  = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, f"cls{cls_id}")

            # Draw box
            bw = 3
            draw.rectangle([x1-bw, y1-bw, x2+bw, y2+bw], outline=(r, g, b), width=bw)
            label = f"{cls_name} {conf:.0%}"
            font_size = max(13, int((x2-x1)*0.08))
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
            tb = draw.textbbox((x1, y1-font_size-6), label, font=font)
            draw.rectangle(tb, fill=(r, g, b))
            draw.text((x1, y1-font_size-6), label, fill=(255,255,255), font=font)

            # Crop plate with small padding
            pad = 4
            cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
            cx2, cy2 = min(rgb.width, x2+pad), min(rgb.height, y2+pad)
            crop = rgb.crop((cx1, cy1, cx2, cy2))

            entries.append({
                "date":       date_str,
                "time":       time_str,
                "source":     source_name,
                "model":      model_short,
                "label":      cls_name,
                "confidence": conf,
                "bbox":       f"({x1},{y1})→({x2},{y2})",
                "size":       f"{x2-x1}×{y2-y1}px",
                "crop_img":   crop,           # PIL Image
                "crop_b64":   img_to_b64(crop),
            })

    return annotated, entries

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "log" not in st.session_state:
    st.session_state.log = []          # list of dicts (one per detected plate)
if "last_annotated" not in st.session_state:
    st.session_state.last_annotated = None
if "last_source" not in st.session_state:
    st.session_state.last_source = None

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;margin-bottom:1.8rem;">
        <div style="font-size:3rem;">🚔</div>
        <div style="font-size:1.15rem;font-weight:800;color:#818cf8;letter-spacing:-0.02em;">
            Vehicle Checker</div>
        <div style="font-size:0.72rem;color:#475569;">License Plate Detection System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🤖 Model")
    selected_model_name = st.selectbox("Model", list(MODELS.keys()),
                                       label_visibility="collapsed")
    model_cfg = MODELS[selected_model_name]
    st.markdown(
        model_cfg["badge"] +
        f'&nbsp;<span style="color:#64748b;font-size:0.78rem;">{model_cfg["description"]}</span>',
        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.95, 0.30, 0.05)
    iou_thresh  = st.slider("IoU (NMS) threshold",  0.10, 0.95, 0.45, 0.05)

    st.markdown("---")
    total = len(st.session_state.log)
    st.metric("Total Plates Logged", total)

    if total > 0 and st.button("🗑️  Clear Log", use_container_width=True):
        st.session_state.log = []
        st.session_state.last_annotated = None
        st.session_state.last_source = None
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#374151;text-align:center;">'
        'bikemodel.pt &bull; carmodel.pt<br>Ultralytics YOLOv8</div>',
        unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
with st.spinner("Loading model…"):
    model = load_model(model_cfg["path"])

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">🚔</div>
    <div>
        <p class="hero-title">Vehicle Checker</p>
        <p class="hero-sub">Detect &amp; log license plates from bikes, cars &amp; trucks in real-time</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input tabs
# ─────────────────────────────────────────────
tab_samples, tab_upload = st.tabs(["🖼️  Sample Images", "📤  Upload Image"])

selected_image: Image.Image | None = None
image_label = ""

with tab_samples:
    is_bike = "Bike" in selected_model_name
    sample_files = BIKE_SAMPLES if is_bike else CAR_SAMPLES

    st.markdown(
        f'<div class="glass-card">Samples for: {model_cfg["badge"]}'
        f'&nbsp;<span style="color:#64748b;font-size:0.8rem;">— click a button to detect</span></div>',
        unsafe_allow_html=True)

    if not sample_files:
        st.warning("No sample images found.")
    else:
        cols_per_row = 4
        rows = [sample_files[i:i+cols_per_row] for i in range(0, len(sample_files), cols_per_row)]
        for row in rows:
            cols = st.columns(len(row))
            for col, fname in zip(cols, row):
                fpath = os.path.join(SAMPLES_DIR, fname)
                try:
                    thumb = Image.open(fpath)
                    with col:
                        st.image(thumb, width="stretch", caption=fname)
                        if st.button("Detect →", key=f"s_{fname}"):
                            selected_image = thumb.copy()
                            image_label    = fname
                except Exception:
                    pass

with tab_upload:
    st.markdown(
        '<div class="glass-card"><b>📂 Upload an image</b><br>'
        '<span style="color:#64748b;font-size:0.82rem;">JPG · JPEG · PNG · WEBP</span></div>',
        unsafe_allow_html=True)
    uploaded = st.file_uploader("Image", type=["jpg","jpeg","png","webp"],
                                label_visibility="collapsed")
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption=uploaded.name, width="stretch")
        if st.button("🔍  Run Detection", use_container_width=True, key="upload_detect"):
            selected_image = img.copy()
            image_label    = uploaded.name

# ─────────────────────────────────────────────
# Run inference
# ─────────────────────────────────────────────
if selected_image is not None:
    with st.spinner("Running detection…"):
        t0 = time.perf_counter()
        annotated, entries = run_inference(
            model, selected_image, conf_thresh, iou_thresh,
            model_cfg["color"], model_cfg["short"], image_label,
        )
        elapsed = time.perf_counter() - t0

    # Append to log
    st.session_state.log.extend(entries)
    st.session_state.last_annotated = annotated
    st.session_state.last_source    = image_label

    st.markdown("---")
    st.markdown("## 🔬 Latest Detection")

    col_orig, col_ann = st.columns(2, gap="large")
    with col_orig:
        st.markdown('<div class="glass-card"><b>📥 Original</b></div>', unsafe_allow_html=True)
        st.image(selected_image, caption=image_label, width="stretch")
    with col_ann:
        st.markdown('<div class="glass-card"><b>📤 Annotated</b></div>', unsafe_allow_html=True)
        st.image(annotated, caption="Detected plates", width="stretch")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Plates Found",    len(entries))
    m2.metric("Inference Time",  f"{elapsed*1000:.0f} ms")
    m3.metric("Confidence Min",  f"{conf_thresh:.0%}")
    m4.metric("Model",           model_cfg["short"])

    if not entries:
        st.markdown(
            '<div class="glass-card" style="text-align:center;color:#94a3b8;">'
            '⚠️ No plates detected — try lowering the confidence threshold.'
            '</div>', unsafe_allow_html=True)

    # Cropped plates preview row
    if entries:
        st.markdown("### 🪪 Cropped Plates from this Image")
        crop_cols = st.columns(min(len(entries), 6))
        for col, e in zip(crop_cols, entries):
            with col:
                st.image(e["crop_img"], caption=f"{e['label']} {e['confidence']:.0%}",
                         width="stretch")

        # Download annotated image
        buf = io.BytesIO()
        annotated.save(buf, format="JPEG", quality=95)
        st.download_button("⬇️  Download Annotated Image", data=buf.getvalue(),
                           file_name=f"detected_{image_label}", mime="image/jpeg",
                           use_container_width=True)

# ─────────────────────────────────────────────
# Detection Log Table
# ─────────────────────────────────────────────
import pandas as pd
import zipfile, csv

st.markdown("---")
st.markdown("## 📋 Detection Log")

log = st.session_state.log

if not log:
    st.markdown(
        '<div class="no-log">🔍 No plates logged yet.<br>'
        '<span style="color:#334155">Run detection on a sample or uploaded image to populate the log.</span></div>',
        unsafe_allow_html=True)
else:
    # Summary metrics
    s1, s2, s3 = st.columns(3)
    s1.metric("Total Entries",   len(log))
    s2.metric("Unique Sources",  len({e["source"] for e in log}))
    s3.metric("Last Detected",   log[-1]["time"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Build DataFrame for st.dataframe ──────
    rows = []
    for i, e in enumerate(reversed(log)):
        rows.append({
            "#":          len(log) - i,
            "Date":       e["date"],
            "Time":       e["time"],
            "Source":     e["source"],
            "Model":      e["model"],
            "Label":      e["label"],
            "Confidence": round(e["confidence"] * 100, 1),
            "BBox":       e["bbox"],
            "Size":       e["size"],
            # st.column_config.ImageColumn accepts data-URI strings
            "Plate Crop": f"data:image/png;base64,{e['crop_b64']}",
        })

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        column_config={
            "Plate Crop": st.column_config.ImageColumn(
                "Plate Crop", help="Cropped license plate region", width="medium"
            ),
            "Confidence": st.column_config.NumberColumn(
                "Confidence (%)", format="%.1f %%"
            ),
            "#": st.column_config.NumberColumn("#", width="small"),
        },
        use_container_width=True,
        hide_index=True,
        height=min(80 + len(rows) * 100, 600),
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Export all crops as ZIP ──────────────
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for i, e in enumerate(log):
            img_buf = io.BytesIO()
            e["crop_img"].save(img_buf, format="PNG")
            zf.writestr(
                f"{e['date']}_{e['time'].replace(':','-')}_plate{i+1}_{e['source']}.png",
                img_buf.getvalue(),
            )
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📦  Export All Crops (ZIP)",
            data=zip_buf.getvalue(),
            file_name="license_plates_log.zip",
            mime="application/zip",
            use_container_width=True,
        )
    with c2:
        # Export CSV (without images)
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf,
            fieldnames=["date","time","source","model","label","confidence","bbox","size"])
        writer.writeheader()
        for e in log:
            writer.writerow({k: e[k] for k in writer.fieldnames})
        st.download_button(
            "📊  Export Log (CSV)",
            data=csv_buf.getvalue(),
            file_name="vehicle_checker_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

