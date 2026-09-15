"""
Streamlit deployment of the NSD tread-depth ensemble. Simple, one-screen
UI for a non-technical user: upload one or more videos, get back the
ensemble tread depth in mm for each.
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
st.markdown('<div class="nsd-sub">Upload one or more scan videos of the tyre tread</div>', unsafe_allow_html=True)

with st.expander("📖 How to record a good video (tap to expand)"):
    st.markdown(
        """
- **Either orientation works** — hold the phone upright (portrait) or sideways (landscape), whichever is comfortable. The app automatically corrects for it.
- Turn the camera **grid on** and set the ratio to **Full** — see the settings example below.
- **Flash/torch ON** for even lighting.
- Keep the camera about **5-8 cm** from the tread.
- **Move slowly left to right** around the tyre — not top to bottom.
- Record for **5-15 seconds**, in HD quality if your phone supports it.
        """
    )
    st.image(
        os.path.join(PROJECT_ROOT, "camera_grid_full_view.jpg"),
        caption="Camera settings: grid on, ratio set to Full",
    )
    st.caption("How to hold and move the phone")
    st.video(os.path.join(PROJECT_ROOT, "How_To_Capture_Video.mp4"))

    col_portrait, col_landscape = st.columns(2)
    with col_portrait:
        st.caption("Example — portrait")
        st.video(os.path.join(PROJECT_ROOT, "sample_video_portrait.mp4"))
    with col_landscape:
        st.caption("Example — landscape")
        st.video(os.path.join(PROJECT_ROOT, "sample_video_landscape.mp4"))


@st.cache_resource(show_spinner="Loading model…")
def load_models():
    fold_checkpoint_paths, fold_stats, n_hand_features = infer._discover_checkpoints()
    device = infer.select_device()
    infer._get_models(fold_checkpoint_paths, n_hand_features, device)
    return fold_checkpoint_paths, fold_stats, n_hand_features, device


fold_checkpoint_paths, fold_stats, n_hand_features, device = load_models()


def run_prediction(video_bytes, suffix):
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


if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "results" not in st.session_state:
    st.session_state.results = []

video_files = st.file_uploader(
    "Choose one or more short videos of the tyre tread",
    type=["mp4", "mov", "avi", "mkv", "m4v"],
    accept_multiple_files=True,
    key=f"video_uploader_{st.session_state.uploader_key}",
)

if video_files:
    if len(video_files) == 1:
        st.video(video_files[0])
    else:
        st.caption(f"{len(video_files)} videos selected: " + ", ".join(f.name for f in video_files))

    col_analyze, col_clear = st.columns([3, 1])
    with col_analyze:
        label = "Analyze video" if len(video_files) == 1 else f"Analyze {len(video_files)} videos"
        analyze_clicked = st.button(label, type="primary", use_container_width=True)
    with col_clear:
        clear_clicked = st.button("Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.uploader_key += 1
        st.session_state.results = []
        st.rerun()

    if analyze_clicked:
        st.session_state.results = []
        progress = st.progress(0.0)
        status = st.empty()
        for i, video_file in enumerate(video_files, start=1):
            status.write(f"Analyzing {video_file.name} ({i}/{len(video_files)})…")
            try:
                suffix = os.path.splitext(video_file.name)[1] or ".mp4"
                result = run_prediction(video_file.getvalue(), suffix)
                st.session_state.results.append({"video": video_file.name, "ensemble_pred_mm": result["ensemble_pred_mm"]})
            except Exception as e:
                st.session_state.results.append({"video": video_file.name, "ensemble_pred_mm": None, "error": str(e)})
            progress.progress(i / len(video_files))
        status.empty()
        progress.empty()
else:
    st.session_state.results = []

results = st.session_state.results

if len(results) == 1 and results[0].get("ensemble_pred_mm") is not None:
    st.markdown(
        f"""
        <div class="nsd-result">
          <div class="label">Estimated Tread Depth</div>
          <div class="value">{results[0]['ensemble_pred_mm']:.2f}<span class="unit"> mm</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif results:
    st.markdown('<div class="nsd-result"><div class="label">Results</div></div>', unsafe_allow_html=True)
    for row in results:
        if row.get("ensemble_pred_mm") is not None:
            st.write(f"**{row['video']}** — {row['ensemble_pred_mm']:.2f} mm")
        else:
            st.write(f"**{row['video']}** — failed: {row['error']}")
