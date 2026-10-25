import streamlit as st
import os
import shutil
import tempfile
import pandas as pd
from PIL import Image

from video_intelligence_engine import VideoIntelligenceEngine
from database import (
    init_db,
    query_detections,
    get_unique_cameras,
    get_detections_for_camera,
    get_all_cameras_meta,
    DEFAULT_DB_PATH
)
from ai_explainer import generate_llm_explanation

# Page setup
st.set_page_config(
    page_title="VisionX · AI Video Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Premium Cyberpunk / Glassmorphism Theme ───────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Base ── */
    .stApp {
        background: #07060b;
        color: #e8e6f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Animated mesh gradient overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse 80% 60% at 10% 20%, rgba(168, 85, 247, 0.12) 0%, transparent 60%),
            radial-gradient(ellipse 60% 70% at 85% 70%, rgba(236, 72, 153, 0.10) 0%, transparent 55%),
            radial-gradient(ellipse 50% 50% at 50% 50%, rgba(56, 189, 248, 0.06) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
        animation: meshPulse 12s ease-in-out infinite alternate;
    }
    @keyframes meshPulse {
        0%   { opacity: 0.7; }
        50%  { opacity: 1;   }
        100% { opacity: 0.8; }
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0a18 0%, #110e1f 50%, #0a0814 100%) !important;
        border-right: 1px solid rgba(168, 85, 247, 0.15) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c084fc !important;
        letter-spacing: 0.03em;
    }

    /* ── Headings ── */
    .stApp h1 {
        background: linear-gradient(135deg, #c084fc 0%, #f472b6 40%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        letter-spacing: -0.02em;
        font-size: 2.2rem !important;
    }
    .stApp h2, .stApp h3 {
        color: #e0d4fc !important;
        font-weight: 700;
    }

    /* ── Glass Cards ── */
    .glass-card {
        background: rgba(17, 14, 31, 0.65);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(168, 85, 247, 0.18);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(192, 132, 252, 0.5), transparent);
    }
    .glass-card:hover {
        border-color: rgba(168, 85, 247, 0.4);
        box-shadow: 0 8px 32px rgba(168, 85, 247, 0.12), 0 0 0 1px rgba(168, 85, 247, 0.08);
        transform: translateY(-2px);
    }

    /* ── Result Photo Card ── */
    .photo-card {
        background: rgba(17, 14, 31, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(168, 85, 247, 0.12);
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 18px;
        transition: all 0.3s ease;
    }
    .photo-card:hover {
        border-color: rgba(244, 114, 182, 0.4);
        box-shadow: 0 12px 40px rgba(244, 114, 182, 0.1);
        transform: translateY(-3px);
    }

    /* ── Pill Badges ── */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.04em;
    }
    .pill-time {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.25) 0%, rgba(139, 92, 246, 0.35) 100%);
        color: #d8b4fe;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .pill-obj {
        background: rgba(56, 189, 248, 0.12);
        color: #7dd3fc;
        border: 1px solid rgba(56, 189, 248, 0.2);
        text-transform: uppercase;
    }
    .pill-score {
        background: rgba(52, 211, 153, 0.12);
        color: #6ee7b7;
        border: 1px solid rgba(52, 211, 153, 0.2);
    }
    .pill-color {
        background: rgba(251, 191, 36, 0.12);
        color: #fcd34d;
        border: 1px solid rgba(251, 191, 36, 0.2);
    }

    /* ── AI Insight Banner ── */
    .ai-insight {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.08) 0%, rgba(244, 114, 182, 0.08) 50%, rgba(56, 189, 248, 0.08) 100%);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 14px;
        padding: 20px 24px;
        margin: 20px 0;
        position: relative;
        overflow: hidden;
    }
    .ai-insight::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #a855f7, #f472b6, #38bdf8);
        border-radius: 4px 0 0 4px;
    }
    .ai-insight h4 {
        margin: 0 0 8px 0;
        background: linear-gradient(90deg, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .ai-insight p {
        margin: 0;
        color: #cbd5e1;
        font-size: 14px;
        line-height: 1.6;
    }

    /* ── Stat Widget ── */
    .stat-box {
        background: rgba(17, 14, 31, 0.55);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(168, 85, 247, 0.12);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .stat-box .stat-value {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .stat-box .stat-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 6px;
        font-weight: 600;
    }

    /* ── Streamlit Overrides ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #a855f7 0%, #d946ef 50%, #ec4899 100%) !important;
        border: none !important;
        color: #fff !important;
        box-shadow: 0 4px 20px rgba(168, 85, 247, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 28px rgba(168, 85, 247, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:not([kind="primary"]) {
        background: rgba(17, 14, 31, 0.6) !important;
        border: 1px solid rgba(168, 85, 247, 0.25) !important;
        color: #c084fc !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        background: rgba(168, 85, 247, 0.12) !important;
        border-color: rgba(168, 85, 247, 0.5) !important;
    }

    /* Text input */
    .stTextInput > div > div > input {
        background: rgba(17, 14, 31, 0.6) !important;
        border: 1px solid rgba(168, 85, 247, 0.25) !important;
        border-radius: 12px !important;
        color: #e8e6f0 !important;
        font-size: 15px !important;
        padding: 12px 16px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.15) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(17, 14, 31, 0.4);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(168, 85, 247, 0.15) !important;
        color: #c084fc !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(17, 14, 31, 0.55);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(168, 85, 247, 0.12);
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] {
        color: #c084fc !important;
        font-weight: 800 !important;
    }

    /* Dividers */
    hr {
        border-color: rgba(168, 85, 247, 0.12) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d0a18; }
    ::-webkit-scrollbar-thumb { background: rgba(168, 85, 247, 0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(168, 85, 247, 0.5); }

    /* Hero subtitle */
    .hero-sub {
        color: #94a3b8;
        font-size: 15px;
        line-height: 1.6;
        max-width: 720px;
    }

    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        animation: statusPulse 2s ease-in-out infinite;
    }
    .status-dot.active { background: #34d399; box-shadow: 0 0 8px rgba(52, 211, 153, 0.5); }
    .status-dot.idle   { background: #fbbf24; box-shadow: 0 0 8px rgba(251, 191, 36, 0.3); }
    @keyframes statusPulse {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.5; }
    }

    /* Chip buttons row */
    .chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }

    /* Dataframe styling */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(17, 14, 31, 0.4) !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "engine" not in st.session_state:
    with st.spinner("⚡ Booting YOLO26 + OpenCLIP neural engines..."):
        st.session_state["engine"] = VideoIntelligenceEngine(yolo_model="yolo26n.pt", clip_model="ViT-B-32")

if "active_catalog" not in st.session_state:
    st.session_state["active_catalog"] = []

if "active_video_path" not in st.session_state:
    st.session_state["active_video_path"] = "videos/gate.mp4" if os.path.exists("videos/gate.mp4") else None

if "show_database" not in st.session_state:
    st.session_state["show_database"] = False

engine = st.session_state["engine"]

# Derive current video name/camera identifier
def get_current_camera_name(vpath):
    if not vpath:
        return "Unknown"
    base = os.path.splitext(os.path.basename(vpath))[0]
    return base.capitalize()

current_camera_name = get_current_camera_name(st.session_state.get("active_video_path"))

# ─── Hero Header ─────────────────────────────────────────────────────────────
st.title("⚡ VisionX · AI Video Intelligence")
st.markdown('<p class="hero-sub">Powered by <strong>YOLO26</strong> real-time detection &amp; <strong>OpenCLIP</strong> multimodal search. Upload any video, describe what you\'re looking for in plain language, and get instant photo evidence with precise timestamps.</p>', unsafe_allow_html=True)

# ─── Sidebar: Video Source ────────────────────────────────────────────────────
st.sidebar.markdown("### 🎞️ Video Source")
video_source_mode = st.sidebar.radio(
    "Choose input:",
    ["📤 Upload Video", "🎥 Sample Footage"],
    label_visibility="collapsed"
)

uploaded_file = None
selected_sample = None

if video_source_mode == "📤 Upload Video":
    uploaded_file = st.sidebar.file_uploader(
        "Drop your video file here",
        type=["mp4", "mov", "avi", "mkv"],
        help="Supported: MP4, MOV, AVI, MKV"
    )
    if uploaded_file is not None:
        temp_dir = os.path.join(tempfile.gettempdir(), "hacknex_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        saved_video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(saved_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # If new video file uploaded
        if st.session_state.get("active_video_path") != saved_video_path:
            st.session_state["active_video_path"] = saved_video_path
            st.session_state["active_catalog"] = []
            st.session_state["show_database"] = False

else:
    sample_options = {
        "🚪 Gate Camera · Vehicles & Pedestrians": "videos/gate.mp4",
        "🅿️ Parking Camera · Cars & Buses": "videos/parking.mp4",
        "🏢 Lobby Camera · Indoor Activity": "videos/lobby.mp4"
    }
    selected_sample = st.sidebar.selectbox("Select feed:", list(sample_options.keys()))
    sample_path = sample_options[selected_sample]
    if os.path.exists(sample_path) and st.session_state.get("active_video_path") != sample_path:
        st.session_state["active_video_path"] = sample_path
        st.session_state["active_catalog"] = []
        st.session_state["show_database"] = False

# Sidebar: Database inspector toggle
st.sidebar.markdown("---")
st.sidebar.markdown("### 🗃️ Data Inspector")
db_sidebar_toggle = st.sidebar.checkbox(
    "Open Database Viewer",
    value=st.session_state["show_database"],
    key="sidebar_db_toggle",
    help="View all SQLite detections, metadata, and timestamps stored for this video"
)
st.session_state["show_database"] = db_sidebar_toggle

# Sidebar: Detection parameters
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Engine Settings")
sample_rate = st.sidebar.slider("Frame sampling interval (sec)", min_value=0.2, max_value=2.0, value=0.5, step=0.1)
yolo_conf = st.sidebar.slider("YOLO26 confidence threshold", min_value=0.15, max_value=0.9, value=0.25, step=0.05)

# Sidebar: Tech stack footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="text-align:center;padding:12px 0;opacity:0.5;font-size:11px;letter-spacing:0.05em;">'
    'YOLO26 · OpenCLIP · SQLite<br>Built for HackNex 2026'
    '</div>',
    unsafe_allow_html=True
)

# ─── Main Content: Video Preview & Status ─────────────────────────────────────
current_video = st.session_state.get("active_video_path")
col_vid, col_status = st.columns([1.3, 1], gap="large")

with col_vid:
    if current_video and os.path.exists(current_video):
        st.video(current_video)
        num_indexed = len(st.session_state["active_catalog"])
        if num_indexed > 0:
            dot_class = "active"
            status_label = f"Indexed · {num_indexed} events"
        else:
            dot_class = "idle"
            status_label = "Awaiting processing"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-top:6px;">'
            f'<span class="status-dot {dot_class}"></span>'
            f'<span style="font-size:13px;color:#94a3b8;font-weight:500;">'
            f'{status_label} · <code>{os.path.basename(current_video)}</code> · Camera: <strong>{current_camera_name}</strong>'
            f'</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:60px 20px;">'
            '<p style="font-size:48px;margin:0;">🎬</p>'
            '<p style="color:#94a3b8;margin-top:12px;">Upload a video or select a sample feed to get started</p>'
            '</div>',
            unsafe_allow_html=True
        )

with col_status:
    st.markdown("### 📡 Index Status")
    num_indexed = len(st.session_state["active_catalog"])

    if num_indexed == 0:
        st.markdown(
            '<div class="glass-card">'
            '<p style="color:#fbbf24;font-weight:600;margin:0 0 8px 0;">⏳ Not Indexed</p>'
            '<p style="color:#94a3b8;font-size:13px;margin:0;line-height:1.5;">Click the button below to process this video through YOLO26 object detection and OpenCLIP visual embeddings.</p>'
            '</div>',
            unsafe_allow_html=True
        )
        if current_video and os.path.exists(current_video):
            if st.button("⚡ Process Video with YOLO26", type="primary", use_container_width=True):
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)

                ev_out_dir = os.path.join("evidence", "uploaded_" + os.path.splitext(os.path.basename(current_video))[0])
                with st.spinner("Running YOLO26 + CLIP neural pipeline..."):
                    cat = engine.process_and_index_video(
                        video_path=current_video,
                        output_dir=ev_out_dir,
                        sample_interval_sec=sample_rate,
                        conf_threshold=yolo_conf,
                        progress_callback=update_progress,
                        camera_name=current_camera_name
                    )
                    st.session_state["active_catalog"] = cat

                st.success(f"✅ Indexed! {len(cat)} detections & keyframes captured.")
                st.rerun()

        if st.button("🗃️ View Database Records", use_container_width=True):
            st.session_state["show_database"] = not st.session_state["show_database"]
            st.rerun()

    else:
        st.markdown(
            f'<div class="glass-card">'
            f'<p style="color:#34d399;font-weight:700;margin:0 0 8px 0;">✅ Video Indexed</p>'
            f'<p style="color:#e0d4fc;font-size:22px;font-weight:800;margin:0;">{num_indexed}</p>'
            f'<p style="color:#94a3b8;font-size:12px;margin:4px 0 0 0;text-transform:uppercase;letter-spacing:0.06em;">Visual events & keyframes in database</p>'
            f'</div>',
            unsafe_allow_html=True
        )
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Re-process", use_container_width=True):
                st.session_state["active_catalog"] = []
                st.rerun()
        with col_btn2:
            db_label = "✖ Close DB" if st.session_state["show_database"] else "🗃️ View DB"
            if st.button(db_label, use_container_width=True):
                st.session_state["show_database"] = not st.session_state["show_database"]
                st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# 🗃️  DATABASE VIEWER
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.get("show_database", False):
    st.markdown("---")
    db_header_col, db_close_col = st.columns([5, 1])
    with db_header_col:
        st.markdown(f"### 🗃️ Database · `{os.path.basename(current_video) if current_video else 'N/A'}`")
        st.caption("Full SQLite record inspection — detections, coordinates, labels, confidence, and metadata.")
    with db_close_col:
        if st.button("✖ Close", use_container_width=True, key="db_close_top"):
            st.session_state["show_database"] = False
            st.rerun()

    # Query detections for this video/camera
    records = get_detections_for_camera(current_camera_name)
    catalog_items = [c for c in st.session_state.get("active_catalog", []) if c.get("type") == "detection"]

    # Stats row
    total_db_dets = len(records)
    total_catalog = len(catalog_items)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("DB Detections", total_db_dets)
    with m2:
        st.metric("Session Indexed", total_catalog)
    with m3:
        unique_classes = len(set(r["object_class"] for r in records)) if records else (len(set(c["object"] for c in catalog_items)) if catalog_items else 0)
        st.metric("Unique Classes", unique_classes)
    with m4:
        st.metric("Storage", os.path.basename(DEFAULT_DB_PATH))

    # Tabs
    tab_table, tab_filter, tab_meta = st.tabs(["📋 Detections", "🔎 Filter & Inspect", "📊 Camera Metadata"])

    with tab_table:
        if records:
            df_rows = []
            for r in records:
                df_rows.append({
                    "ID": r.get("id"),
                    "Camera": r.get("camera"),
                    "Timestamp": r.get("timestamp_str"),
                    "Time (s)": round(r.get("timestamp_sec", 0), 2),
                    "Class": r.get("object_class"),
                    "Confidence": f"{float(r.get('confidence', 0)):.2%}",
                    "Color": r.get("color_hint") or "—",
                    "BBox": str(r.get("bbox", [])),
                    "Evidence": r.get("evidence_path")
                })
            df = pd.DataFrame(df_rows)
            st.dataframe(df, use_container_width=True, height=350)

            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export CSV",
                data=csv_data,
                file_name=f"{current_camera_name}_detections.csv",
                mime="text/csv"
            )
        elif catalog_items:
            st.info("Showing in-memory catalog (current session):")
            df_rows = []
            for idx, c in enumerate(catalog_items):
                df_rows.append({
                    "#": idx + 1,
                    "Camera": current_camera_name,
                    "Timestamp": c.get("timestamp_str"),
                    "Time (s)": c.get("timestamp_sec"),
                    "Class": c.get("object"),
                    "Confidence": f"{float(c.get('confidence', 0)):.2%}",
                    "Color": c.get("color") or "—",
                    "BBox": str(c.get("bbox", [])),
                    "Evidence": c.get("evidence_path")
                })
            df = pd.DataFrame(df_rows)
            st.dataframe(df, use_container_width=True, height=350)
        else:
            st.warning(f"No records for '{current_camera_name}'. Process the video first.")

    with tab_filter:
        if records or catalog_items:
            source_data = records if records else [
                {"object_class": c["object"], "confidence": c["confidence"], "timestamp_sec": c["timestamp_sec"], "timestamp_str": c["timestamp_str"], "color_hint": c.get("color"), "bbox": c.get("bbox"), "evidence_path": c.get("evidence_path")}
                for c in catalog_items
            ]
            all_classes = sorted(list(set(r.get("object_class") for r in source_data if r.get("object_class"))))

            fc1, fc2 = st.columns(2)
            with fc1:
                selected_cls = st.selectbox("Object class:", ["All"] + all_classes)
            with fc2:
                min_conf = st.slider("Min confidence:", 0.0, 1.0, 0.25, 0.05)

            filtered = [
                r for r in source_data
                if (selected_cls == "All" or r.get("object_class") == selected_cls) and float(r.get("confidence", 0)) >= min_conf
            ]
            st.write(f"**{len(filtered)}** matching detections")

            if filtered:
                preview_cols = st.columns(min(4, max(1, len(filtered))))
                for idx, f_item in enumerate(filtered[:4]):
                    with preview_cols[idx % len(preview_cols)]:
                        ev_path = f_item.get("evidence_path", "")
                        if ev_path and os.path.exists(ev_path):
                            st.image(ev_path, caption=f"{f_item.get('object_class')} @ {f_item.get('timestamp_str')}")
                        else:
                            st.caption(f"{f_item.get('object_class')} @ {f_item.get('timestamp_str')}")

    with tab_meta:
        all_cams = get_all_cameras_meta()
        if all_cams:
            cam_df = pd.DataFrame(all_cams)
            st.write("Registered video streams:")
            st.dataframe(cam_df, use_container_width=True)
        else:
            st.info("No cameras registered yet.")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# 🔎  NATURAL LANGUAGE SEARCH
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### 🔎 Describe What You're Looking For")
st.markdown('<p class="hero-sub">Type anything in natural language — <em>"Find the car"</em>, <em>"person in black shirt"</em>, <em>"white vehicle near gate"</em></p>', unsafe_allow_html=True)

col_search, col_btn = st.columns([5, 1])
with col_search:
    user_query = st.text_input(
        "Search",
        value=st.session_state.get("last_query", "Find a car"),
        placeholder="Describe what you're looking for...",
        label_visibility="collapsed"
    )
with col_btn:
    execute_search = st.button("🔍 Search", type="primary", use_container_width=True)

# Quick-search chips
st.markdown('<div style="margin-top:4px;">', unsafe_allow_html=True)
chip_cols = st.columns(5)
chip_queries = [
    ("🚗", "Find a car"),
    ("🚶", "Find a person"),
    ("🚲", "Show bicycle"),
    ("⚪", "White vehicle"),
    ("🎒", "Person with bag"),
]
for i, (emoji, q) in enumerate(chip_queries):
    with chip_cols[i]:
        if st.button(f"{emoji} {q}", use_container_width=True, key=f"chip_{i}"):
            user_query = q
            execute_search = True
st.markdown('</div>', unsafe_allow_html=True)

# Execute Search
if (execute_search or user_query) and len(st.session_state["active_catalog"]) > 0:
    st.session_state["last_query"] = user_query

    with st.spinner(f"Searching for: '{user_query}'..."):
        results = engine.search_video(
            user_query=user_query,
            catalog=st.session_state["active_catalog"],
            top_k=9
        )

    # AI Insight Banner
    if results:
        top = results[0]
        color_info = f" ({top.get('color', '')})" if top.get('color') and top['color'] != 'unknown' else ""
        st.markdown(f"""
        <div class="ai-insight">
            <h4>🎯 AI Detection Result</h4>
            <p>
                Found <strong>{top['object']}{color_info}</strong> at timestamp 
                <strong>{top['timestamp_str']}</strong> ({top['timestamp_sec']}s) 
                with <strong>{int(top['match_score']*100)}%</strong> match confidence. 
                Identified <strong>{len(results)} distinct moments</strong> across this video.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"No matches found for '{user_query}' in this video.")

    st.markdown(f"### 📸 Results · {len(results)} matches")

    # Photo grid
    if results:
        cols = st.columns(3, gap="medium")
        for idx, item in enumerate(results):
            col = cols[idx % 3]
            with col:
                color_pill = ""
                if item.get('color') and item['color'] != 'unknown':
                    color_pill = f'<span class="pill pill-color">{item["color"]}</span>'

                st.markdown(f"""
                <div class="photo-card">
                    <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;">
                        <span class="pill pill-time">⏱ {item['timestamp_str']}</span>
                        <span class="pill pill-obj">{item['object']}</span>
                        <span class="pill pill-score">⚡ {int(item['match_score']*100)}%</span>
                        {color_pill}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Photo Evidence
                if os.path.exists(item["evidence_path"]):
                    img = Image.open(item["evidence_path"])
                    st.image(img, use_container_width=True)
                else:
                    st.caption(f"📷 {item['evidence_path']}")

                # Play from timestamp
                if current_video and os.path.exists(current_video):
                    with st.expander(f"▶ Play from {item['timestamp_str']}"):
                        st.video(current_video, start_time=int(item["timestamp_sec"]))

elif len(st.session_state["active_catalog"]) == 0 and current_video:
    st.markdown(
        '<div class="glass-card" style="text-align:center;padding:32px;">'
        '<p style="font-size:36px;margin:0;">🧠</p>'
        '<p style="color:#94a3b8;margin-top:10px;">Process the video first to unlock natural language search</p>'
        '</div>',
        unsafe_allow_html=True
    )
