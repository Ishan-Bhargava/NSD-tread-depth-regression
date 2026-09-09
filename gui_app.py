"""
Desktop GUI for the NSD (tread depth) ResNet18-hybrid 5-fold ensemble.

Wraps nsd_inference_attention_pcr.py (kept completely untouched apart from
making CHECKPOINT_DIR resolve relative to this folder) in a Tkinter /
ttkbootstrap window: pick a video, run the ensemble, see the prediction.

Packaged into a single .exe with PyInstaller -- see build_exe.py /
nsd_gui.spec.
"""

import os
import sys
import queue
import threading
import traceback
from datetime import datetime

import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_DND = True
except Exception:
    _HAS_DND = False

# When frozen by PyInstaller, sys.executable's folder is where bundled data
# (checkpoints, fold_stats.json) gets unpacked to (see nsd_gui.spec).
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)
import nsd_inference_attention_pcr as infer

# Make sure the ensemble finds its checkpoints both in dev (running this
# .py directly) and once frozen by PyInstaller, where they're unpacked
# under sys._MEIPASS rather than next to this source file.
infer.CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoint_1600_data_attention_excluded_2to9")
infer.FOLD_STATS_PATH = os.path.join(infer.CHECKPOINT_DIR, "fold_stats.json")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv")

APP_TITLE = "NSD Tread Depth Estimator"
THEME = "flatly"


def fmt_mm(value):
    return f"{value:.2f} mm"


class NSDApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("980x680")
        self.root.minsize(880, 620)

        self.style = tb.Style(theme=THEME)

        self.video_path = None
        self.msg_queue = queue.Queue()
        self.models_ready = False
        self.fold_checkpoint_paths = None
        self.fold_stats = None
        self.n_hand_features = None
        self.device = None
        self.history = []

        self._build_layout()
        self.root.after(100, self._poll_queue)

        threading.Thread(target=self._load_models_bg, daemon=True).start()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        header = tb.Frame(self.root, padding=(24, 18, 24, 12))
        header.pack(fill=X)

        tb.Label(header, text=APP_TITLE, font=("Segoe UI", 20, "bold")).pack(anchor=W)
        tb.Label(
            header,
            text="ResNet18-hybrid attention model • 5-fold ensemble • runs fully offline",
            font=("Segoe UI", 10),
            bootstyle="secondary",
        ).pack(anchor=W, pady=(2, 0))

        self.status_var = tb.StringVar(value="Loading model checkpoints…")
        self.status_lbl = tb.Label(header, textvariable=self.status_var, font=("Segoe UI", 9), bootstyle="secondary")
        self.status_lbl.pack(anchor=W, pady=(8, 0))

        body = tb.Frame(self.root, padding=(24, 0, 24, 18))
        body.pack(fill=BOTH, expand=YES)
        body.columnconfigure(0, weight=5)
        body.columnconfigure(1, weight=6)
        body.rowconfigure(0, weight=1)

        self._build_left_panel(body)
        self._build_right_panel(body)

    def _build_left_panel(self, parent):
        left = tb.Labelframe(parent, text=" Video ", padding=16)
        left.grid(row=0, column=0, sticky=NSEW, padx=(0, 12))

        self.drop_zone = tb.Frame(left, bootstyle="light")
        self.drop_zone.pack(fill=X, pady=(4, 12))
        self.drop_zone.configure(height=150)

        inner = tb.Frame(self.drop_zone)
        inner.place(relx=0.5, rely=0.5, anchor="center")
        tb.Label(inner, text="\U0001F3A5", font=("Segoe UI", 30)).pack()
        drop_hint = "Drag & drop a video here, or" if _HAS_DND else "Select a tyre inspection video"
        tb.Label(inner, text=drop_hint, font=("Segoe UI", 10), bootstyle="secondary").pack(pady=(4, 8))
        tb.Button(inner, text="Browse…", command=self._browse_file, bootstyle="primary").pack()

        if _HAS_DND:
            self.drop_zone.drop_target_register(DND_FILES)
            self.drop_zone.dnd_bind("<<Drop>>", self._on_drop)

        self.file_lbl = tb.Label(left, text="No video selected", font=("Segoe UI", 10, "bold"), wraplength=380)
        self.file_lbl.pack(fill=X, pady=(0, 4))
        self.file_meta_lbl = tb.Label(left, text="", font=("Segoe UI", 9), bootstyle="secondary")
        self.file_meta_lbl.pack(fill=X, pady=(0, 16))

        self.run_btn = tb.Button(
            left, text="Run Inference", command=self._run_inference, bootstyle="success", state=DISABLED,
        )
        self.run_btn.pack(fill=X, pady=(0, 10))

        self.progress = tb.Progressbar(left, mode="indeterminate", bootstyle="success-striped")
        self.progress.pack(fill=X, pady=(0, 6))

        self.run_status_lbl = tb.Label(left, text="", font=("Segoe UI", 9), bootstyle="secondary")
        self.run_status_lbl.pack(fill=X)

        tb.Separator(left).pack(fill=X, pady=14)

        tb.Label(left, text="Recent predictions", font=("Segoe UI", 10, "bold")).pack(anchor=W, pady=(0, 6))
        hist_frame = tb.Frame(left)
        hist_frame.pack(fill=BOTH, expand=YES)
        columns = ("video", "ensemble", "std", "time")
        self.hist_tree = tb.Treeview(
            hist_frame, columns=columns, show="headings", height=8, bootstyle="light",
        )
        self.hist_tree.heading("video", text="Video")
        self.hist_tree.heading("ensemble", text="Ensemble")
        self.hist_tree.heading("std", text="Fold std")
        self.hist_tree.heading("time", text="Time")
        self.hist_tree.column("video", width=150, anchor=W)
        self.hist_tree.column("ensemble", width=80, anchor=CENTER)
        self.hist_tree.column("std", width=70, anchor=CENTER)
        self.hist_tree.column("time", width=70, anchor=CENTER)
        self.hist_tree.pack(fill=BOTH, expand=YES)
        self.hist_tree.bind("<<TreeviewSelect>>", self._on_history_select)

    def _build_right_panel(self, parent):
        right = tb.Labelframe(parent, text=" Prediction ", padding=16)
        right.grid(row=0, column=1, sticky=NSEW)

        self.result_card = tb.Frame(right, bootstyle="light")
        self.result_card.pack(fill=X, pady=(4, 16))
        card_inner = tb.Frame(self.result_card, padding=20)
        card_inner.pack(fill=X)

        tb.Label(card_inner, text="ENSEMBLE TREAD DEPTH", font=("Segoe UI", 10, "bold"), bootstyle="secondary").pack(anchor=W)
        self.big_result_var = tb.StringVar(value="—")
        self.big_result_lbl = tb.Label(card_inner, textvariable=self.big_result_var, font=("Segoe UI", 44, "bold"), bootstyle="success")
        self.big_result_lbl.pack(anchor=W)

        self.badge_var = tb.StringVar(value="")
        self.badge_lbl = tb.Label(card_inner, textvariable=self.badge_var, font=("Segoe UI", 10, "bold"))
        self.badge_lbl.pack(anchor=W, pady=(4, 0))

        tb.Label(right, text="Per-fold breakdown", font=("Segoe UI", 10, "bold")).pack(anchor=W, pady=(0, 8))
        self.bars_canvas = tb.Canvas(right, height=190, highlightthickness=0)
        self.bars_canvas.pack(fill=X, pady=(0, 16))
        self.bars_canvas.bind("<Configure>", lambda e: self._redraw_bars())

        self.detail_txt = tb.Text(right, height=8, font=("Consolas", 9), wrap="word")
        self.detail_txt.pack(fill=BOTH, expand=YES)
        self.detail_txt.configure(state=DISABLED)

        self._last_result = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_models_bg(self):
        try:
            fold_checkpoint_paths, fold_stats, n_hand_features = infer._discover_checkpoints()
            device = infer.select_device()
            models = infer._get_models(fold_checkpoint_paths, n_hand_features, device)
            self.fold_checkpoint_paths = fold_checkpoint_paths
            self.fold_stats = fold_stats
            self.n_hand_features = n_hand_features
            self.device = device
            self.msg_queue.put(("models_ready", (len(models), str(device))))
        except Exception as e:
            self.msg_queue.put(("models_error", str(e)))

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------
    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select a tyre inspection video",
            filetypes=[("Video files", " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)), ("All files", "*.*")],
        )
        if path:
            self._set_selected_file(path)

    def _on_drop(self, event):
        raw = event.data.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        path = raw.split("} {")[0] if "} {" in raw else raw
        if os.path.splitext(path)[1].lower() not in VIDEO_EXTENSIONS:
            messagebox.showwarning(APP_TITLE, "That doesn't look like a supported video file.")
            return
        self._set_selected_file(path)

    def _set_selected_file(self, path):
        self.video_path = path
        size_mb = os.path.getsize(path) / (1024 * 1024)
        self.file_lbl.configure(text=os.path.basename(path))
        self.file_meta_lbl.configure(text=f"{size_mb:.1f} MB")
        if self.models_ready:
            self.run_btn.configure(state=NORMAL)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _run_inference(self):
        if not self.video_path:
            return
        self.run_btn.configure(state=DISABLED)
        self.progress.start(12)
        self.run_status_lbl.configure(text="Sampling frames & running ensemble…")
        threading.Thread(target=self._run_inference_bg, args=(self.video_path,), daemon=True).start()

    def _run_inference_bg(self, video_path):
        try:
            result = infer.predict_video(
                video_path,
                fold_checkpoint_paths=self.fold_checkpoint_paths,
                fold_stats=self.fold_stats,
                n_hand_features=self.n_hand_features,
                device=self.device,
            )
            self.msg_queue.put(("inference_done", result))
        except Exception as e:
            self.msg_queue.put(("inference_error", (str(e), traceback.format_exc())))

    # ------------------------------------------------------------------
    # Queue polling (keeps all widget mutation on the Tk main thread)
    # ------------------------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "models_ready":
                    n_folds, device = payload
                    self.models_ready = True
                    self.status_var.set(f"Model ready — {n_folds} fold(s) loaded • running on {device.upper() if device=='cpu' else device}")
                    if self.video_path:
                        self.run_btn.configure(state=NORMAL)
                elif kind == "models_error":
                    self.status_var.set("Failed to load model checkpoints — see error dialog")
                    messagebox.showerror(APP_TITLE, f"Could not load model checkpoints:\n\n{payload}")
                elif kind == "inference_done":
                    self._on_inference_done(payload)
                elif kind == "inference_error":
                    err, tb_str = payload
                    self.progress.stop()
                    self.run_status_lbl.configure(text="Failed — see error dialog")
                    self.run_btn.configure(state=NORMAL)
                    print(tb_str)
                    messagebox.showerror(APP_TITLE, f"Inference failed:\n\n{err}")
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    def _on_inference_done(self, result):
        self.progress.stop()
        self.run_btn.configure(state=NORMAL)
        self.run_status_lbl.configure(text="Done.")
        self._last_result = result
        self._show_result(result)
        self._add_history(result)

    # ------------------------------------------------------------------
    # Result rendering
    # ------------------------------------------------------------------
    def _show_result(self, result):
        ensemble = result["ensemble_pred_mm"]
        std = result["fold_pred_std_mm"]
        self.big_result_var.set(fmt_mm(ensemble))

        if std < 0.15:
            badge_text, style = f"Folds agree closely (±{std:.2f} mm)", "success"
        elif std < 0.4:
            badge_text, style = f"Moderate fold disagreement (±{std:.2f} mm)", "warning"
        else:
            badge_text, style = f"High fold disagreement (±{std:.2f} mm) — treat with caution", "danger"
        self.badge_var.set(badge_text)
        self.badge_lbl.configure(bootstyle=style)
        self.big_result_lbl.configure(bootstyle=style)

        self._redraw_bars()

        fold_keys = sorted(k for k in result if k.startswith("pred_fold"))
        lines = [f"Video: {os.path.basename(result['video_path'])}", ""]
        for k in fold_keys:
            fold_n = k.replace("pred_fold", "").replace("_mm", "")
            lines.append(f"  Fold {fold_n}: {result[k]:.3f} mm")
        lines.append("")
        lines.append(f"Ensemble mean : {ensemble:.3f} mm")
        lines.append(f"Fold std      : {std:.3f} mm")
        self.detail_txt.configure(state=NORMAL)
        self.detail_txt.delete("1.0", "end")
        self.detail_txt.insert("1.0", "\n".join(lines))
        self.detail_txt.configure(state=DISABLED)

    def _redraw_bars(self):
        self.bars_canvas.delete("all")
        if not self._last_result:
            return
        result = self._last_result
        fold_keys = sorted(k for k in result if k.startswith("pred_fold"))
        values = [result[k] for k in fold_keys]
        labels = [k.replace("pred_fold", "Fold ").replace("_mm", "") for k in fold_keys]

        w = self.bars_canvas.winfo_width() or 500
        h = self.bars_canvas.winfo_height() or 190
        if w < 10:
            return
        pad_left, pad_right, pad_top, pad_bottom = 70, 60, 10, 10
        plot_w = max(w - pad_left - pad_right, 50)
        plot_h = h - pad_top - pad_bottom
        n = len(values)
        if n == 0:
            return
        bar_h = plot_h / n * 0.55
        gap = plot_h / n

        vmax = max(values + [result["ensemble_pred_mm"]]) * 1.15 or 1.0
        vmin = min(0, min(values) * 0.9)
        span = vmax - vmin or 1.0

        ens_x = pad_left + (result["ensemble_pred_mm"] - vmin) / span * plot_w
        self.bars_canvas.create_line(ens_x, pad_top - 2, ens_x, pad_top + plot_h + 2, dash=(4, 2), fill="#6c757d")

        for i, (label, val) in enumerate(zip(labels, values)):
            y0 = pad_top + i * gap + (gap - bar_h) / 2
            y1 = y0 + bar_h
            x0 = pad_left
            x1 = pad_left + (val - vmin) / span * plot_w
            self.bars_canvas.create_text(pad_left - 10, (y0 + y1) / 2, text=label, anchor="e", font=("Segoe UI", 9))
            self.bars_canvas.create_rectangle(x0, y0, x1, y1, fill="#2fa4e7", outline="")
            self.bars_canvas.create_text(x1 + 6, (y0 + y1) / 2, text=f"{val:.2f}", anchor="w", font=("Segoe UI", 9, "bold"))

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    def _add_history(self, result):
        entry = dict(result)
        entry["_timestamp"] = datetime.now().strftime("%H:%M:%S")
        self.history.append(entry)
        self.hist_tree.insert(
            "", 0,
            iid=str(len(self.history) - 1),
            values=(
                os.path.basename(result["video_path"]),
                fmt_mm(result["ensemble_pred_mm"]),
                f"±{result['fold_pred_std_mm']:.2f}",
                entry["_timestamp"],
            ),
        )

    def _on_history_select(self, _event):
        sel = self.hist_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        self._last_result = self.history[idx]
        self._show_result(self.history[idx])


def main():
    root = TkinterDnD.Tk() if _HAS_DND else tb.Window()
    if _HAS_DND:
        tb.Style(theme=THEME)
    NSDApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
