"""
Streamlit deployment of the NSD tread-depth ensemble. Simple, one-screen
UI for a non-technical user: upload a video, get back one number -- the
ensemble tread depth in mm.
"""

import os
import sys
import tempfile

import streamlit as st

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
st.markdown('<div class="nsd-sub">Upload a scan video of the tyre tread</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model…")
def load_models():
    fold_checkpoint_paths, fold_stats, n_hand_features = infer._discover_checkpoints()
    device = infer.select_device()
    infer._get_models(fold_checkpoint_paths, n_hand_features, device)
    return fold_checkpoint_paths, fold_stats, n_hand_features, device


fold_checkpoint_paths, fold_stats, n_hand_features, device = load_models()


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


result = None

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
