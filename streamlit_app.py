"""
Streamlit deployment of the NSD tread-depth ensemble. Simple, one-screen
UI for a non-technical user: pick a video OR record one with the camera,
get back one number -- the ensemble tread depth in mm.

Camera recording uses a small custom component (components/video_recorder)
that records a real video clip in the browser via getUserMedia +
MediaRecorder and hands the result back as base64 -- st.camera_input only
captures a single still photo, which isn't enough for this model's
frame-sampling pipeline.
"""

import base64
import os
import sys
import tempfile

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import nsd_inference_attention_pcr as infer

infer.CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoint_1600_data_attention_excluded_2to9")
infer.FOLD_STATS_PATH = os.path.join(infer.CHECKPOINT_DIR, "fold_stats.json")
infer.CACHE_DIR = os.path.join(tempfile.gettempdir(), "nsd_inference_cache")

st.set_page_config(page_title="Tyre Tread Depth (NSD)", page_icon="🛞", layout="centered")

st.markdown("""
<style>
  #MainMenu, footer, header {visibility: hidden;}
  .block-container {padding-top: 2.5rem; max-width: 480px;}
  .nsd-title {text-align: center; font-size: 1.6rem; font-weight: 800; margin-bottom: 0;}
  .nsd-sub {text-align: center; color: #6b7480; font-size: 0.9rem; margin-bottom: 1.6rem;}
  .nsd-result {text-align: center; padding: 18px 0 6px;}
  .nsd-result .label {color: #6b7480; font-size: 0.8rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;}
  .nsd-result .value {font-size: 3.4rem; font-weight: 800; color: #16a34a; margin: 6px 0 0;}
  .nsd-result .unit {color: #6b7480; font-size: 1rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="nsd-title">🛞 Tyre Tread Depth (NSD)</div>', unsafe_allow_html=True)
st.markdown('<div class="nsd-sub">Upload a scan video, or record one with your camera</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model…")
def load_models():
    fold_checkpoint_paths, fold_stats, n_hand_features = infer._discover_checkpoints()
    device = infer.select_device()
    infer._get_models(fold_checkpoint_paths, n_hand_features, device)
    return fold_checkpoint_paths, fold_stats, n_hand_features, device


fold_checkpoint_paths, fold_stats, n_hand_features, device = load_models()

_video_recorder = components.declare_component(
    "video_recorder", path=os.path.join(PROJECT_ROOT, "components", "video_recorder"),
)


def video_recorder(key=None):
    return _video_recorder(key=key, default=None)


def run_prediction(video_bytes, suffix):
    with st.spinner("Analyzing tread depth…"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        try:
            return infer.predict_video(
                tmp_path, fold_checkpoint_paths, fold_stats, n_hand_features, device,
                use_cache=False,
            )
        finally:
            os.remove(tmp_path)


tab_video, tab_camera = st.tabs(["📁 Upload video", "🎥 Record video"])

result = None

with tab_video:
    video_file = st.file_uploader(
        "Choose a short video of the tyre tread", type=["mp4", "mov", "avi", "mkv", "m4v"],
    )
    if video_file is not None:
        st.video(video_file)
        if st.button("Analyze video", type="primary", use_container_width=True):
            try:
                suffix = os.path.splitext(video_file.name)[1] or ".mp4"
                result = run_prediction(video_file.getvalue(), suffix)
            except Exception as e:
                st.error(f"Could not analyze this video: {e}")

with tab_camera:
    st.caption("Allow camera access, record a few seconds panning across the tread, then send it.")
    recorder_value = video_recorder(key="cam_recorder")
    if recorder_value and recorder_value.get("video_b64"):
        if st.button("Analyze recording", type="primary", use_container_width=True, key="analyze_cam"):
            try:
                mime = recorder_value.get("mime_type", "video/webm")
                suffix = ".mp4" if "mp4" in mime else ".webm"
                video_bytes = base64.b64decode(recorder_value["video_b64"])
                result = run_prediction(video_bytes, suffix)
            except Exception as e:
                st.error(f"Could not analyze this recording: {e}")

if result is not None:
    st.markdown(
        f"""
        <div class="nsd-result">
          <div class="label">Estimated Tread Depth</div>
          <div class="value">{result['ensemble_pred_mm']:.2f}<span class="unit"> mm</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
