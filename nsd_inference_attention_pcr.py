"""
Standalone inference script for the ResNet18-hybrid tire tread depth
model. FULLY SELF-CONTAINED -- no dependency on pcr_training_lib.py.

Why: this typically runs on a different machine (local Windows box) than
training (Kaggle). Keeping two separate files in sync turned out to be a
recurring source of "Missing key(s) in state_dict" architecture-mismatch
errors whenever the training script changed. Every preprocessing function
and the model architecture are reproduced directly in this single file.

IMPORTANT -- READ THIS BEFORE YOUR NEXT TRAINING RUN CHANGES ANYTHING:
If you change FRAMES_PER_VIDEO, IMG_SIZE, ATTN_MAP_SIZE, the hand-crafted
feature extraction, the attention-target computation, the orientation
normalization, or the model architecture in pcr_training_lib.py for a
FUTURE training run, you must manually mirror those same changes here too
-- there is no automatic sync anymore, on purpose (that indirection was
the actual cause of the errors you hit). Two different failure modes if
these drift apart:
  - Architecture mismatch -> an immediate, loud "Missing/unexpected
    key(s) in state_dict" crash when loading a checkpoint.
  - Preprocessing mismatch (e.g. a different FRAMES_PER_VIDEO, or
    forgetting orientation normalization) -> NO crash, just silently
    worse/wrong predictions -- much harder to notice than a crash, so
    double check this file against pcr_training_lib.py's CONFIG section
    and preprocessing functions any time you change either one.

USAGE
=====
    import pcr_inference as infer

    result = infer.predict_video('/path/to/video.mp4')
    print(result['ensemble_pred_mm'])

    df = infer.predict_folder('/path/to/videos_dir', output_csv='predictions.csv')

    df = infer.predict_from_csv('videos.csv', output_csv='predictions.csv')

Or just set the CONFIG variables below and run this file directly.
"""

import os
import json
import glob

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# =========================
# CONFIG -- EDIT THESE
# =========================

# Where your trained fold checkpoints + fold_stats.json actually live.
# Resolved relative to this file so the script/app works no matter where
# the project folder is checked out.
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoint_1600_data_attention_excluded_2to9')
FOLD_STATS_PATH = os.path.join(CHECKPOINT_DIR, 'fold_stats.json')

# What to run inference on: a single video, a folder of videos, or a CSV
# with a 'video_path' or 'filename' column.
INPUT_PATH = r'D:\testing_image\nsd_pcr_2300\video_976.mp4'
VIDEOS_DIR = None  # only used if INPUT_PATH is a CSV with a 'filename' column, not 'video_path'
OUTPUT_CSV_PATH = 'inference_predictions.csv'

# Local on-disk cache of decoded frames/features, so re-scoring the same
# video twice is fast the second time. Safe to delete any time -- gets
# rebuilt from the raw video on the next run.
CACHE_DIR = 'inference_cache'

# Must match pcr_training_lib.py's CONFIG exactly -- see module docstring.
FRAMES_PER_VIDEO = 16
FRAME_OVERSAMPLE_MULTIPLIER = 3
IMG_SIZE = 224
ATTN_MAP_SIZE = 7  # must match the CNN backbone's final spatial map (7x7 for resnet18 @ 224x224)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def select_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    cap_major, cap_minor = torch.cuda.get_device_capability(0)
    cap_str = f"sm_{cap_major}{cap_minor}"
    if cap_str not in torch.cuda.get_arch_list():
        print(f"Warning: GPU compute capability {cap_str} not supported by this PyTorch build -- using CPU.")
        return torch.device('cpu')
    return torch.device('cuda')


# =========================
# PREPROCESSING -- must match pcr_training_lib.py exactly
# =========================

def extract_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    f1 = float(np.std(gray))
    edges = cv2.Canny(gray, 50, 150)
    f2 = float(np.sum(edges > 0) / edges.size)
    blur = cv2.GaussianBlur(gray, (31, 31), 0)
    depth_map = blur.astype(np.float32) - gray.astype(np.float32)
    f3 = float(np.percentile(depth_map, 90))
    f4 = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    f5 = float(np.mean(np.abs(sobelx)))
    return [f1, f2, f3, f4, f5]


def compute_attention_target_map(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray_eq, 50, 150).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray_eq, (31, 31), 0)
    groove = np.abs(blur.astype(np.float32) - gray_eq.astype(np.float32))
    groove_norm = groove / (groove.max() + 1e-6)
    combined = 0.5 * edges + 0.5 * groove_norm
    return combined


def frame_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def ensure_vertical_frame(frame):
    """Rotates landscape frames to portrait -- must match training (see
    pcr_training_lib.py's module docstring on orientation normalization
    and why it matters for feature f5 / shortcut-learning risk)."""
    h, w = frame.shape[:2]
    if w > h:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def sample_best_frames(video_path, n_frames, oversample_multiplier=FRAME_OVERSAMPLE_MULTIPLIER):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(ensure_vertical_frame(frame))
        cap.release()
        if not frames:
            return []
        scored = [(i, f, frame_sharpness(f)) for i, f in enumerate(frames)]
        scored.sort(key=lambda x: x[2], reverse=True)
        top = sorted(scored[:n_frames], key=lambda x: x[0])
        return [f for _, f, _ in top]

    n_candidates = min(total, max(n_frames, n_frames * oversample_multiplier))
    candidate_idxs = np.linspace(0, total - 1, n_candidates).astype(int)

    candidates = []
    for idx in candidate_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            candidates.append((int(idx), ensure_vertical_frame(frame)))
    cap.release()

    if not candidates:
        return []

    scored = [(idx, frame, frame_sharpness(frame)) for idx, frame in candidates]
    scored.sort(key=lambda x: x[2], reverse=True)
    top = sorted(scored[:n_frames], key=lambda x: x[0])
    return [frame for _, frame, _ in top]


def build_attention_targets(attn_target_full, attn_map_size=ATTN_MAP_SIZE):
    T = attn_target_full.shape[0]
    small = np.stack([
        cv2.resize(attn_target_full[t].astype(np.float32), (attn_map_size, attn_map_size), interpolation=cv2.INTER_AREA)
        for t in range(T)
    ])
    small = np.clip(small, 0, None)

    flat = small.reshape(T, -1)
    sums = flat.sum(axis=1, keepdims=True)
    zero_mask = (sums.flatten() <= 1e-6)

    safe_sums = np.where(sums <= 1e-6, 1.0, sums)
    target_dist = flat / safe_sums
    if zero_mask.any():
        target_dist[zero_mask] = 1.0 / (attn_map_size * attn_map_size)
    target_dist = target_dist.reshape(T, attn_map_size, attn_map_size)

    raw_score = flat.sum(axis=1).astype(np.float32)
    score_min, score_max = raw_score.min(), raw_score.max()
    if score_max - score_min < 1e-6:
        frame_quality = np.zeros(T, dtype=np.float32)
    else:
        frame_quality = (raw_score - score_min) / (score_max - score_min)

    return target_dist.astype(np.float32), frame_quality


def frames_to_tensor(resized_frames):
    x = resized_frames.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (0, 3, 1, 2))
    return torch.from_numpy(x).float().unsqueeze(0)


# =========================
# LOCAL CACHE
# Simplified vs. pcr_training_lib.py's version (no read-only/multi-source
# directory logic) -- this runs standalone on one machine, one cache dir.
# =========================

def _cache_paths(cache_key):
    return (
        os.path.join(CACHE_DIR, f"{cache_key}_orientfix_n{FRAMES_PER_VIDEO}_s{IMG_SIZE}.npy"),
        os.path.join(CACHE_DIR, f"{cache_key}_orientfix_handfeats.npy"),
        os.path.join(CACHE_DIR, f"{cache_key}_orientfix_attntarget.npy"),
    )


def _frames_to_data(frames):
    """Shared feature/attention/resize pipeline for a list of already
    oriented BGR frames -- used for both a sampled video and a single
    still photo (repeated FRAMES_PER_VIDEO times, see get_image_data)."""
    hand_features = np.median([extract_features(f) for f in frames], axis=0).astype(np.float32)

    resized = np.stack([
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
        for f in frames
    ])

    attn_targets = np.stack([
        (cv2.resize(compute_attention_target_map(f), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA) * 255)
        .clip(0, 255).astype(np.uint8)
        for f in frames
    ])

    if resized.shape[0] < FRAMES_PER_VIDEO:
        pad = np.repeat(resized[-1:], FRAMES_PER_VIDEO - resized.shape[0], axis=0)
        resized = np.concatenate([resized, pad], axis=0)
        attn_pad = np.repeat(attn_targets[-1:], FRAMES_PER_VIDEO - attn_targets.shape[0], axis=0)
        attn_targets = np.concatenate([attn_targets, attn_pad], axis=0)

    return resized, hand_features, attn_targets


def get_video_data(video_path, use_cache=True):
    """Returns (resized_frames, hand_features, attn_target_full), using
    the on-disk cache when available. Set use_cache=False to force
    re-decoding even if a cache entry already exists under this filename
    -- useful if a filename got reused for a different physical video,
    which would otherwise silently serve stale cached data."""
    cache_key = os.path.splitext(os.path.basename(video_path))[0]
    frames_p, feats_p, attn_p = _cache_paths(cache_key)

    if use_cache and os.path.exists(frames_p) and os.path.exists(feats_p) and os.path.exists(attn_p):
        return np.load(frames_p), np.load(feats_p), np.load(attn_p)

    frames = sample_best_frames(video_path, FRAMES_PER_VIDEO)
    if len(frames) == 0:
        return None, None, None

    resized, hand_features, attn_targets = _frames_to_data(frames)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(frames_p, resized)
        np.save(feats_p, hand_features)
        np.save(attn_p, attn_targets)

    return resized, hand_features, attn_targets


def get_image_data(image_bgr):
    """Same (resized_frames, hand_features, attn_target_full) triple as
    get_video_data, but for a single still photo (e.g. a phone/webcam
    snapshot) instead of a video -- no temporal diversity to pick a best
    frame from, so the one frame is just repeated FRAMES_PER_VIDEO times.
    Not cached: stills are one-off, unlike re-scoring the same video."""
    frame = ensure_vertical_frame(image_bgr)
    frames = [frame] * FRAMES_PER_VIDEO
    return _frames_to_data(frames)


# =========================
# MODEL ARCHITECTURE -- must match pcr_training_lib.py exactly
# =========================

class SpatialAttentionPool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        attn_logits = self.attn_conv(x)
        N, _, H, W = attn_logits.shape
        attn_weights = torch.softmax(attn_logits.view(N, -1), dim=-1).view(N, 1, H, W)
        pooled = (x * attn_weights).sum(dim=[2, 3])
        return pooled, attn_weights


class TemporalAttentionPool(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, frame_quality):
        x_with_quality = torch.cat([x, frame_quality.unsqueeze(-1)], dim=-1)
        scores = self.attn(x_with_quality).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1)
        return pooled, weights


class ResNetHybridRegressor(nn.Module):
    def __init__(self, n_hand_features, pretrained=False):
        super().__init__()
        # pretrained=False here -- at inference time every weight gets
        # overwritten by the trained checkpoint anyway, so there's no
        # reason to download ImageNet weights first.
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.feature_dim = backbone.fc.in_features

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.spatial_attn_pool = SpatialAttentionPool(self.feature_dim)
        self.temporal_attn_pool = TemporalAttentionPool(self.feature_dim)

        combined_dim = self.feature_dim + n_hand_features
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, frames, hand_features, frame_quality, return_attention=False):
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        pooled_spatial, spatial_attn = self.spatial_attn_pool(x)
        pooled_spatial = pooled_spatial.view(B, T, self.feature_dim)

        pooled_temporal, temporal_attn = self.temporal_attn_pool(pooled_spatial, frame_quality)

        combined = torch.cat([pooled_temporal, hand_features], dim=1)
        out = self.head(combined).squeeze(1)

        if return_attention:
            H_map, W_map = spatial_attn.shape[-2], spatial_attn.shape[-1]
            spatial_attn = spatial_attn.view(B, T, 1, H_map, W_map)
            return out, spatial_attn, temporal_attn
        return out


def load_fold_model(checkpoint_path, n_hand_features, device):
    model = ResNetHybridRegressor(n_hand_features=n_hand_features, pretrained=False).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# =========================
# CHECKPOINT DISCOVERY
# =========================

def _discover_checkpoints(checkpoint_dir=None, fold_stats_path=None):
    """Loads fold_stats.json, matches it against whichever
    resnet_hybrid_depth_fold{N}_best.pt files actually exist on disk, and
    derives n_hand_features from fold_stats.json's own 'feature_names'
    entry (rather than a separately hardcoded constant here) -- one less
    place for a count mismatch to silently creep in."""
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    fold_stats_path = fold_stats_path or FOLD_STATS_PATH

    if not os.path.exists(fold_stats_path):
        raise FileNotFoundError(
            f"'{fold_stats_path}' not found -- set CHECKPOINT_DIR to wherever your trained "
            f"checkpoints + fold_stats.json actually live."
        )
    with open(fold_stats_path) as f:
        fold_stats_json = json.load(f)

    feature_names = fold_stats_json.get("feature_names")
    if not feature_names:
        raise ValueError(f"'{fold_stats_path}' has no 'feature_names' entry -- unexpected format.")
    n_hand_features = len(feature_names)

    fold_checkpoint_paths = {}
    fold_stats = {}
    for key, stats in fold_stats_json.items():
        if key == "feature_names":
            continue
        fold_idx = int(key)
        checkpoint_path = os.path.join(checkpoint_dir, f"resnet_hybrid_depth_fold{fold_idx}_best.pt")
        if not os.path.exists(checkpoint_path):
            print(f"  Warning: fold_stats.json references fold {fold_idx} but "
                  f"'{checkpoint_path}' doesn't exist on disk -- skipping.")
            continue
        fold_checkpoint_paths[fold_idx] = checkpoint_path
        fold_stats[fold_idx] = {
            "mean": np.array(stats["mean"], dtype=np.float32),
            "std": np.array(stats["std"], dtype=np.float32),
        }

    if not fold_checkpoint_paths:
        raise FileNotFoundError(f"No usable checkpoints found under '{checkpoint_dir}' matching fold_stats.json.")
    print(f"Loaded {len(fold_checkpoint_paths)} fold checkpoint(s): {sorted(fold_checkpoint_paths)}")
    return fold_checkpoint_paths, fold_stats, n_hand_features


_MODELS_CACHE = {}  # avoids reloading + re-uploading the same checkpoints to GPU repeatedly


def _get_models(fold_checkpoint_paths, n_hand_features, device):
    key = (tuple(sorted(fold_checkpoint_paths.items())), n_hand_features, str(device))
    if key in _MODELS_CACHE:
        return _MODELS_CACHE[key]
    models = {
        fold_idx: load_fold_model(path, n_hand_features, device)
        for fold_idx, path in fold_checkpoint_paths.items()
    }
    _MODELS_CACHE[key] = models
    return models


# =========================
# PUBLIC API
# =========================

def _run_ensemble(resized_frames, hand_features, attn_target_full,
                   fold_checkpoint_paths, fold_stats, n_hand_features, device):
    """Runs the 5-fold ensemble on one already-preprocessed
    (resized_frames, hand_features, attn_target_full) triple. Returns a
    dict with each fold's prediction, the ensemble average, and
    fold_pred_std_mm (how much the folds disagree -- a free, cheap
    uncertainty signal). Shared by predict_video and predict_image."""
    frames_tensor = frames_to_tensor(resized_frames).to(device)
    _, frame_quality = build_attention_targets(attn_target_full)
    frame_quality_tensor = torch.from_numpy(frame_quality).float().unsqueeze(0).to(device)

    models = _get_models(fold_checkpoint_paths, n_hand_features, device)

    result = {}
    fold_preds = []
    for fold_idx, model in models.items():
        mean = fold_stats[fold_idx]["mean"]
        std = fold_stats[fold_idx]["std"]
        normalized = (hand_features - mean) / std
        hand_features_tensor = torch.from_numpy(normalized).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(frames_tensor, hand_features_tensor, frame_quality_tensor).item()
        result[f"pred_fold{fold_idx}_mm"] = pred
        fold_preds.append(pred)

    result["ensemble_pred_mm"] = float(np.mean(fold_preds))
    result["fold_pred_std_mm"] = float(np.std(fold_preds))
    return result


def predict_video(video_path, fold_checkpoint_paths=None, fold_stats=None, n_hand_features=None,
                   device=None, use_cache=True):
    """Runs the 5-fold ensemble on ONE video. No ground-truth depth needed
    or used anywhere. Returns a dict with each fold's prediction, the
    ensemble average, and fold_pred_std_mm (how much the folds disagree --
    a free, cheap uncertainty signal)."""
    if fold_checkpoint_paths is None or fold_stats is None or n_hand_features is None:
        fold_checkpoint_paths, fold_stats, n_hand_features = _discover_checkpoints()
    if device is None:
        device = select_device()

    resized_frames, hand_features, attn_target_full = get_video_data(video_path, use_cache=use_cache)
    if resized_frames is None:
        raise ValueError(f"Could not extract any frames from '{video_path}'.")

    result = _run_ensemble(resized_frames, hand_features, attn_target_full,
                            fold_checkpoint_paths, fold_stats, n_hand_features, device)
    result["video_path"] = video_path
    result["unique_id"] = os.path.splitext(os.path.basename(video_path))[0]
    return result


def predict_image(image_bgr, fold_checkpoint_paths=None, fold_stats=None, n_hand_features=None, device=None):
    """Runs the 5-fold ensemble on a single still photo (e.g. a phone/
    webcam snapshot) instead of a video -- see get_image_data for the
    tradeoff this implies (no best-frame selection, flat frame_quality)."""
    if fold_checkpoint_paths is None or fold_stats is None or n_hand_features is None:
        fold_checkpoint_paths, fold_stats, n_hand_features = _discover_checkpoints()
    if device is None:
        device = select_device()

    resized_frames, hand_features, attn_target_full = get_image_data(image_bgr)

    result = _run_ensemble(resized_frames, hand_features, attn_target_full,
                            fold_checkpoint_paths, fold_stats, n_hand_features, device)
    return result


def predict_folder(videos_dir, output_csv=None, extensions=(".mp4", ".mov", ".avi", ".mkv")):
    """Runs inference on every video file in videos_dir (non-recursive)."""
    fold_checkpoint_paths, fold_stats, n_hand_features = _discover_checkpoints()
    device = select_device()
    print(f"Running inference on device: {device}")

    video_files = sorted([
        f for f in glob.glob(os.path.join(videos_dir, "*"))
        if os.path.splitext(f)[1].lower() in extensions
    ])
    if not video_files:
        raise FileNotFoundError(f"No video files found in '{videos_dir}' with extensions {extensions}.")
    print(f"Found {len(video_files)} video(s) in '{videos_dir}'.")

    rows = []
    for i, video_path in enumerate(video_files, start=1):
        try:
            result = predict_video(video_path, fold_checkpoint_paths, fold_stats, n_hand_features, device)
            rows.append(result)
            print(f"  [{i}/{len(video_files)}] {result['unique_id']}: "
                  f"ensemble={result['ensemble_pred_mm']:.3f}mm (fold std={result['fold_pred_std_mm']:.3f})")
        except Exception as e:
            print(f"  [{i}/{len(video_files)}] {os.path.basename(video_path)}: FAILED -- {e}")

    df = pd.DataFrame(rows)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nSaved {len(df)} prediction(s) to '{output_csv}'.")
    return df


def predict_from_csv(csv_path, videos_dir=None, output_csv=None):
    """Reads a CSV with a 'video_path' column (full paths) or a 'filename'
    column (joined with videos_dir). Extra columns in the input (e.g. a
    known depth_mm, if re-scoring held_out_test_set.csv to sanity check
    this script) are carried through to the output untouched, purely for
    your own reference -- never read or used by the model."""
    df_in = pd.read_csv(csv_path)
    if "video_path" in df_in.columns:
        paths = df_in["video_path"].astype(str).tolist()
    elif "filename" in df_in.columns:
        if videos_dir is None:
            raise ValueError("CSV has a 'filename' column but no videos_dir was given to join it with.")
        paths = [os.path.join(videos_dir, f) for f in df_in["filename"].astype(str)]
    else:
        raise ValueError(f"'{csv_path}' needs a 'video_path' or 'filename' column.")

    fold_checkpoint_paths, fold_stats, n_hand_features = _discover_checkpoints()
    device = select_device()
    print(f"Running inference on device: {device}")
    print(f"Scoring {len(paths)} video(s) from '{csv_path}'.")

    rows = []
    for i, path in enumerate(paths, start=1):
        try:
            result = predict_video(path, fold_checkpoint_paths, fold_stats, n_hand_features, device)
            rows.append(result)
            print(f"  [{i}/{len(paths)}] {result['unique_id']}: ensemble={result['ensemble_pred_mm']:.3f}mm")
        except Exception as e:
            print(f"  [{i}/{len(paths)}] {os.path.basename(str(path))}: FAILED -- {e}")
            rows.append({
                "video_path": path,
                "unique_id": os.path.splitext(os.path.basename(str(path)))[0],
                "error": str(e),
            })

    df_out = pd.DataFrame(rows)
    if len(df_out) == len(df_in):
        for col in df_in.columns:
            if col not in df_out.columns:
                df_out[col] = df_in[col].values

    if output_csv:
        df_out.to_csv(output_csv, index=False)
        print(f"\nSaved {len(df_out)} prediction(s) to '{output_csv}'.")
    return df_out


def main():
    print(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")
    print(f"INPUT_PATH: {INPUT_PATH}")
    print(f"OUTPUT_CSV_PATH: {OUTPUT_CSV_PATH}")

    if os.path.isdir(INPUT_PATH):
        predict_folder(INPUT_PATH, output_csv=OUTPUT_CSV_PATH)
    elif INPUT_PATH.lower().endswith(".csv"):
        predict_from_csv(INPUT_PATH, videos_dir=VIDEOS_DIR, output_csv=OUTPUT_CSV_PATH)
    else:
        fold_checkpoint_paths, fold_stats, n_hand_features = _discover_checkpoints()
        device = select_device()
        result = predict_video(INPUT_PATH, fold_checkpoint_paths, fold_stats, n_hand_features, device)
        print(f"\nEnsemble prediction: {result['ensemble_pred_mm']:.1f} mm "
              f"(fold disagreement std: {result['fold_pred_std_mm']:.3f} mm)")
        pd.DataFrame([result]).to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Saved to '{OUTPUT_CSV_PATH}'.")


if __name__ == "__main__":
    main()