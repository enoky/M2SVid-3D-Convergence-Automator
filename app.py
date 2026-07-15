import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import re
import csv
import json
import cv2
import numpy as np
import threading
import subprocess
import torch
import torch.nn.functional as F
from convergence_estimator import ConvergenceEstimator
from depth_scaler import EMAMinMaxScaler

OUTPUT_VIDEO_NAME = "M2SVid_Convergence_Control.mp4"
OUTPUT_CSV_NAME = "M2SVid_Convergence_Control.csv"
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi')

# The control video encodes a single scalar per frame and the Fusion Probe node
# samples one pixel, so output resolution is irrelevant — a tiny frame keeps
# encoding near-instant and files small. Only the frame rate must match.
CONTROL_SIZE = 64

# U2NETP input resolution. Frames are downscaled to this immediately after
# decode so the sync/scaler queues hold small tensors instead of full frames.
MODEL_INPUT_SIZE = 192

INFERENCE_BATCH_SIZE = 8

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")


class ProcessingStopped(Exception):
    """Raised internally when the user presses Stop."""
    pass


def natural_sort_key(name):
    """Sort helper so scene2 < scene10; os.listdir order is not guaranteed."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


def fps_to_rate_string(fps):
    """Map decoded fps to an exact FFmpeg rate, preserving NTSC fractional rates."""
    ntsc_rates = (
        (24000 / 1001, "24000/1001"),
        (30000 / 1001, "30000/1001"),
        (48000 / 1001, "48000/1001"),
        (60000 / 1001, "60000/1001"),
        (120000 / 1001, "120000/1001"),
    )
    for target, rate in ntsc_rates:
        if abs(fps - target) < 0.005:
            return rate
    if abs(fps - round(fps)) < 1e-6:
        return str(int(round(fps)))
    return f"{fps:.6f}"


class VideoProcessorApp:
    """
    GUI application for generating automated convergence control videos
    for stereoscopic 3D movies created with M2SVid.

    Takes synchronized RGB video + matching depth video as input,
    uses the ConvergenceEstimator model to predict per-frame convergence
    values, applies EMA smoothing + optional temporal filtering, and
    outputs a grayscale video (0-255 brightness = convergence value).

    The resulting video is designed to be imported into DaVinci Resolve
    as a control track to automatically drive the Convergence parameter
    throughout the movie.
    """
    def __init__(self, root):
        """
        Initializes the main application window and its widgets.
        """
        self.root = root
        self.root.title("M2SVid 3D Convergence Automator")
        self.root.geometry("700x720")
        self.root.minsize(620, 660)

        self.stop_event = threading.Event()
        self.estimator = None

        # --- Style Configuration ---
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure("TLabel", padding=6, font=("Helvetica", 10))
        style.configure("TButton", padding=6, font=("Helvetica", 10, "bold"))
        style.configure("TEntry", padding=6, font=("Helvetica", 10))
        style.configure("TFrame", padding=10)
        style.configure("Header.TLabel", font=("Helvetica", 14, "bold"))

        # --- Main Frame ---
        main_frame = ttk.Frame(self.root, padding=(10, 10, 10, 10))
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(1, weight=1)

        # --- Title ---
        header_label = ttk.Label(main_frame, text="M2SVid 3D Convergence Automator", style="Header.TLabel")
        header_label.grid(row=0, column=0, columnspan=3, pady=(0, 20), sticky="w")

        # --- Folder Selection ---
        ttk.Label(main_frame, text="Input (RGB) Folder:").grid(row=1, column=0, sticky="w")
        self.input_folder_var = tk.StringVar(value="No folder selected")
        self.input_folder_label = ttk.Label(main_frame, textvariable=self.input_folder_var, wraplength=450, foreground="gray")
        self.input_folder_label.grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(main_frame, text="Browse...",
                   command=lambda: self._select_folder("Select Input Folder", self.input_folder_var, self.input_folder_label)
                   ).grid(row=1, column=2, sticky="e")

        ttk.Label(main_frame, text="Input (Depth) Folder:").grid(row=2, column=0, sticky="w")
        self.depth_folder_var = tk.StringVar(value="No folder selected")
        self.depth_folder_label = ttk.Label(main_frame, textvariable=self.depth_folder_var, wraplength=450, foreground="gray")
        self.depth_folder_label.grid(row=2, column=1, sticky="ew", padx=5)
        ttk.Button(main_frame, text="Browse...",
                   command=lambda: self._select_folder("Select Depth Folder", self.depth_folder_var, self.depth_folder_label)
                   ).grid(row=2, column=2, sticky="e")

        ttk.Label(main_frame, text="Output Folder:").grid(row=3, column=0, sticky="w")
        self.output_folder_var = tk.StringVar(value="No folder selected")
        self.output_folder_label = ttk.Label(main_frame, textvariable=self.output_folder_var, wraplength=450, foreground="gray")
        self.output_folder_label.grid(row=3, column=1, sticky="ew", padx=5)
        ttk.Button(main_frame, text="Browse...",
                   command=lambda: self._select_folder("Select Output Folder", self.output_folder_var, self.output_folder_label)
                   ).grid(row=3, column=2, sticky="e")

        # Frame for Convergence Ratio and EMA
        self.convergence_frame = ttk.Frame(main_frame)
        self.convergence_frame.grid(row=4, column=0, columnspan=3, sticky='ew', pady=(15, 0))
        ttk.Label(self.convergence_frame, text="Convergence Ratio (0.0 - 1.0):").grid(row=0, column=0, sticky="w")
        self.convergence_ratio_var = tk.StringVar(value="0.6")
        self.convergence_ratio_entry = ttk.Entry(self.convergence_frame, textvariable=self.convergence_ratio_var, width=10)
        self.convergence_ratio_entry.grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(self.convergence_frame, text=" | EMA Alpha (0.01 - 1.0):").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.ema_alpha_var = tk.StringVar(value="0.2")
        self.ema_alpha_entry = ttk.Entry(self.convergence_frame, textvariable=self.ema_alpha_var, width=10)
        self.ema_alpha_entry.grid(row=0, column=3, sticky="w", padx=5)

        ttk.Label(self.convergence_frame, text="Scaler Decay (0.0 - 1.0):").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.scaler_decay_var = tk.StringVar(value="0.9")
        self.scaler_decay_entry = ttk.Entry(self.convergence_frame, textvariable=self.scaler_decay_var, width=10)
        self.scaler_decay_entry.grid(row=1, column=1, sticky="w", padx=5, pady=(10, 0))

        ttk.Label(self.convergence_frame, text=" | Scaler Buffer (frames):").grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(10, 0))
        self.scaler_buffer_var = tk.StringVar(value="30")
        self.scaler_buffer_entry = ttk.Entry(self.convergence_frame, textvariable=self.scaler_buffer_var, width=10)
        self.scaler_buffer_entry.grid(row=1, column=3, sticky="w", padx=5, pady=(10, 0))

        ttk.Label(self.convergence_frame, text="Temporal Window (frames):").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.temporal_window_var = tk.StringVar(value="31")
        self.temporal_window_entry = ttk.Entry(self.convergence_frame, textvariable=self.temporal_window_var, width=10)
        self.temporal_window_entry.grid(row=2, column=1, sticky="w", padx=5, pady=(10, 0))

        self.cross_clip_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.convergence_frame, text="Smooth across clip boundaries",
                        variable=self.cross_clip_var).grid(row=2, column=2, columnspan=2, sticky="w", padx=(10, 0), pady=(10, 0))

        # --- Preview Row ---
        preview_frame = ttk.Frame(main_frame)
        preview_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Label(preview_frame, text="Preview seconds:").grid(row=0, column=0, sticky="w")
        self.preview_seconds_var = tk.StringVar(value="10")
        ttk.Entry(preview_frame, textvariable=self.preview_seconds_var, width=10).grid(row=0, column=1, sticky="w", padx=5)
        self.preview_button = ttk.Button(preview_frame, text="Preview Curve (first clip)", command=self.start_preview_thread)
        self.preview_button.grid(row=0, column=2, sticky="w", padx=(10, 0))

        # --- Progress Bar and Status ---
        self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal", mode="determinate")
        self.progress_bar.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(20, 5))

        self.status_label = ttk.Label(main_frame, text="Waiting to start...")
        self.status_label.grid(row=7, column=0, columnspan=3, sticky="ew", padx=5)

        # --- Log/Status Display ---
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding=10)
        log_frame.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(10, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(8, weight=1)

        self.log_text = tk.Text(log_frame, height=10, state="disabled", wrap="word", bg="#f0f0f0", relief="sunken", borderwidth=1)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=scrollbar.set)

        # --- Action Button ---
        self.start_button = ttk.Button(main_frame, text="Start Processing", command=self.start_processing_thread)
        self.start_button.grid(row=9, column=0, columnspan=3, pady=(10, 0), sticky="ew")

        self._load_settings()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Logging / status helpers (thread-safe via root.after)
    # ------------------------------------------------------------------
    def _log_message(self, message):
        self.root.after(0, self._update_log_text, message)

    def _update_log_text(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.config(state="disabled")
        self.log_text.see(tk.END)

    def _set_status(self, message):
        self.root.after(0, self.status_label.config, {'text': message})

    def _set_progress(self, value):
        self.root.after(0, self.progress_bar.config, {'value': value})

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------
    def _settings_vars(self):
        return {
            'input_folder': self.input_folder_var,
            'depth_folder': self.depth_folder_var,
            'output_folder': self.output_folder_var,
            'convergence_ratio': self.convergence_ratio_var,
            'ema_alpha': self.ema_alpha_var,
            'scaler_decay': self.scaler_decay_var,
            'scaler_buffer': self.scaler_buffer_var,
            'temporal_window': self.temporal_window_var,
            'cross_clip': self.cross_clip_var,
            'preview_seconds': self.preview_seconds_var,
        }

    def _load_settings(self):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        for key, var in self._settings_vars().items():
            if key in data:
                try:
                    var.set(data[key])
                except tk.TclError:
                    pass
        for var, label in ((self.input_folder_var, self.input_folder_label),
                           (self.depth_folder_var, self.depth_folder_label),
                           (self.output_folder_var, self.output_folder_label)):
            if os.path.isdir(var.get()):
                label.config(foreground="black")

    def _save_settings(self):
        try:
            data = {key: var.get() for key, var in self._settings_vars().items()}
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except (OSError, tk.TclError):
            pass

    def _on_close(self):
        self.stop_event.set()
        self._save_settings()
        self.root.destroy()

    # ------------------------------------------------------------------
    # Folder selection
    # ------------------------------------------------------------------
    def _select_folder(self, title, var, label):
        folder_path = filedialog.askdirectory(title=title)
        if folder_path:
            var.set(folder_path)
            label.config(foreground="black")

    # ------------------------------------------------------------------
    # Parameter parsing / run state
    # ------------------------------------------------------------------
    def _collect_params(self):
        """Parse and validate the tunable parameters. Returns dict or None."""
        params = {}
        try:
            params['convergence_ratio'] = float(self.convergence_ratio_var.get())
            if not (0.0 <= params['convergence_ratio'] <= 1.0): raise ValueError
            params['ema_alpha'] = float(self.ema_alpha_var.get())
            if not (0.01 <= params['ema_alpha'] <= 1.0): raise ValueError
            params['scaler_decay'] = float(self.scaler_decay_var.get())
            if not (0.0 <= params['scaler_decay'] <= 1.0): raise ValueError
            params['scaler_buffer'] = int(self.scaler_buffer_var.get())
            if params['scaler_buffer'] < 1: raise ValueError
            params['temporal_window'] = int(self.temporal_window_var.get())
            if params['temporal_window'] < 1: raise ValueError
            if params['temporal_window'] % 2 == 0:
                params['temporal_window'] += 1  # Enforce odd number for a perfectly centered window
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters. Please check ratio, decay (floats 0-1), alpha (0.01-1) and buffers (int >= 1).")
            return None
        params['cross_clip'] = bool(self.cross_clip_var.get())
        return params

    def _set_busy_state(self):
        self.start_button.config(text="Stop", command=self.request_stop, state="normal")
        self.preview_button.config(state="disabled")

    def _set_idle_state(self):
        self.start_button.config(text="Start Processing", command=self.start_processing_thread, state="normal")
        self.preview_button.config(state="normal")

    def request_stop(self):
        self.stop_event.set()
        self.start_button.config(state="disabled", text="Stopping...")
        self._set_status("Stopping...")

    def start_processing_thread(self):
        input_folder = self.input_folder_var.get()
        depth_folder = self.depth_folder_var.get()
        output_folder = self.output_folder_var.get()

        if not os.path.isdir(input_folder) or not os.path.isdir(output_folder):
            messagebox.showerror("Error", "Please select valid input and output folders.")
            return

        if not os.path.isdir(depth_folder):
            messagebox.showerror("Error", "Please select a valid depth folder for the Convergence Model.")
            return

        params = self._collect_params()
        if params is None:
            return
        params['depth_folder'] = depth_folder

        self._save_settings()
        self.stop_event.clear()
        self._set_busy_state()
        self._set_progress(0)

        processing_thread = threading.Thread(
            target=self.process_videos_controller,
            args=(input_folder, output_folder, params),
            daemon=True
        )
        processing_thread.start()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def start_preview_thread(self):
        input_folder = self.input_folder_var.get()
        depth_folder = self.depth_folder_var.get()
        if not os.path.isdir(input_folder) or not os.path.isdir(depth_folder):
            messagebox.showerror("Error", "Please select valid input (RGB) and depth folders.")
            return

        params = self._collect_params()
        if params is None:
            return
        try:
            preview_seconds = float(self.preview_seconds_var.get())
            if preview_seconds <= 0: raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Preview seconds must be a positive number.")
            return
        params['depth_folder'] = depth_folder

        self._save_settings()
        self.stop_event.clear()
        self._set_busy_state()
        threading.Thread(target=self._run_preview, args=(input_folder, params, preview_seconds), daemon=True).start()

    def _run_preview(self, input_dir, params, preview_seconds):
        try:
            video_files = sorted(
                (f for f in os.listdir(input_dir) if f.lower().endswith(VIDEO_EXTENSIONS)),
                key=natural_sort_key)
            if not video_files:
                self._log_message("No video files found in the input directory.")
                self.root.after(0, lambda: messagebox.showwarning("Warning", "No video files found."))
                return

            filename = video_files[0]
            rgb_path = os.path.join(input_dir, filename)
            base_name, ext = os.path.splitext(filename)
            depth_path = os.path.join(params['depth_folder'], f"{base_name}_depth{ext}")
            if not os.path.exists(depth_path):
                self._log_message(f"  [!] Error: Depth video not found for {filename}.")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Depth video not found for {filename}."))
                return

            cap = cv2.VideoCapture(rgb_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps <= 0:
                self._log_message(f"  [!] Error: Could not read fps from {filename}.")
                return

            max_frames = max(int(round(preview_seconds * fps)), 1)
            self._log_message(f"\n--- Preview: first {preview_seconds:g}s of '{filename}' ({max_frames} frames) ---")
            result = self._process_predict(rgb_path, depth_path, filename, params, max_frames=max_frames)
            if result is None:
                return
            values, _ = result
            values = self._apply_box_filter(values, params['temporal_window'])
            self._log_message(f"  > Preview complete: {len(values)} frames analyzed.")
            self._set_status("Preview complete.")
            self.root.after(0, self._show_curve_window, values, filename, params)
        except ProcessingStopped:
            self._log_message("\n--- Preview stopped by user ---")
            self._set_status("Stopped.")
        except Exception as e:
            self._log_message(f"Preview error: {e}")
        finally:
            self.root.after(0, self._set_idle_state)

    def _show_curve_window(self, values, filename, params):
        """Plot the convergence curve on a plain tk Canvas (no extra deps)."""
        if not values:
            return
        win = tk.Toplevel(self.root)
        win.title(f"Convergence Curve — {filename}")
        w, h = 680, 340
        margin_l, margin_r, margin_t, margin_b = 45, 15, 15, 30
        canvas = tk.Canvas(win, width=w, height=h, bg="white", highlightthickness=0)
        canvas.pack(padx=10, pady=(10, 0))
        plot_w = w - margin_l - margin_r
        plot_h = h - margin_t - margin_b

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = margin_t + plot_h * (1.0 - frac)
            canvas.create_line(margin_l, y, w - margin_r, y, fill="#dddddd")
            canvas.create_text(margin_l - 6, y, text=f"{frac:.2f}", anchor="e", font=("Helvetica", 8))
        canvas.create_text(margin_l, h - margin_b + 6, text="0", anchor="nw", font=("Helvetica", 8))
        canvas.create_text(w - margin_r, h - margin_b + 6, text=str(len(values) - 1), anchor="ne", font=("Helvetica", 8))
        canvas.create_text(w // 2, h - margin_b + 6, text="frame", anchor="n", font=("Helvetica", 8))

        if len(values) > 1:
            points = []
            for idx, v in enumerate(values):
                x = margin_l + plot_w * idx / (len(values) - 1)
                y = margin_t + plot_h * (1.0 - min(max(v, 0.0), 1.0))
                points.extend((x, y))
            canvas.create_line(*points, fill="#1f77b4", width=1.5)

        stats = (f"frames={len(values)}  min={min(values):.3f}  max={max(values):.3f}  "
                 f"mean={sum(values) / len(values):.3f}\n"
                 f"ratio={params['convergence_ratio']}  alpha={params['ema_alpha']}  "
                 f"decay={params['scaler_decay']}  buffer={params['scaler_buffer']}  "
                 f"window={params['temporal_window']}")
        ttk.Label(win, text=stats, justify="center").pack(padx=10, pady=(4, 10))

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------
    def process_videos_controller(self, input_dir, output_dir, params):
        try:
            self._log_message("--- Starting Process ---")
            self._log_message("Mode: Convergence Model (predict)")
            video_files = sorted(
                (f for f in os.listdir(input_dir) if f.lower().endswith(VIDEO_EXTENSIONS)),
                key=natural_sort_key)
            if not video_files:
                self._log_message("No video files found in the input directory.")
                self.root.after(0, lambda: messagebox.showwarning("Warning", "No video files found."))
                return

            self._log_message(f"Clip order: {', '.join(video_files)}")

            # Validate every RGB/depth pair up front. Skipping a clip mid-run
            # would silently desync everything after it, so refuse to start
            # until the inputs are consistent.
            self._set_status("Validating clips...")
            clips, problems = self._prescan_clips(input_dir, params['depth_folder'], video_files)
            if problems:
                for p in problems:
                    self._log_message(f"  [!] {p}")
                self._log_message("Aborting: fix the issues above and run again "
                                  "(skipping clips would desync the control track).")
                summary = "\n".join(problems[:10])
                self.root.after(0, lambda: messagebox.showerror("Validation failed", summary))
                return

            self.root.after(0, self.progress_bar.config, {'maximum': len(clips) * 2})

            # --- Analysis pass ---
            clip_values = []  # list of (clip_info, [float 0-1 per frame])
            prev_val = None
            for i, clip in enumerate(clips):
                self._log_message(f"\nProcessing '{clip['filename']}' ({i + 1}/{len(clips)})...")
                carry = prev_val if params['cross_clip'] else None
                result = self._process_predict(clip['rgb_path'], clip['depth_path'],
                                               clip['filename'], params, initial_prev_val=carry)
                if result is None:
                    self._log_message(f"  [!] Analysis failed; writing neutral gray for "
                                      f"{clip['frame_count']} frames to preserve sync.")
                    clip_values.append((clip, [0.5] * clip['frame_count']))
                    prev_val = None
                else:
                    values, prev_val = result
                    clip_values.append((clip, values))
                self._set_progress(i + 1)

            # --- Temporal box filtering ---
            window = params['temporal_window']
            if params['cross_clip'] and window > 1:
                self._log_message(f"\n> Applying temporal smoothing across all clips (window={window})...")
                all_values = [v for _, values in clip_values for v in values]
                all_smoothed = self._apply_box_filter(all_values, window)
                offset = 0
                smoothed = []
                for clip, values in clip_values:
                    smoothed.append((clip, all_smoothed[offset:offset + len(values)]))
                    offset += len(values)
                clip_values = smoothed
            elif window > 1:
                self._log_message(f"\n> Applying per-clip temporal smoothing (window={window})...")
                clip_values = [(clip, self._apply_box_filter(values, window))
                               for clip, values in clip_values]

            clip_brightness = [
                (clip, [int(round(min(max(v, 0.0), 1.0) * 255)) for v in values])
                for clip, values in clip_values
            ]

            # --- Write pass ---
            fps_str = fps_to_rate_string(clips[0]['fps'])
            if not self._write_output(output_dir, fps_str, clip_brightness, len(clips)):
                return
            self._write_csv(output_dir, clip_values, clip_brightness)

            self._log_message("\n--- Processing Complete ---")
            self._set_status("Finished.")
            self.root.after(0, lambda: messagebox.showinfo(
                "Success",
                f"All videos have been processed successfully!\n\n"
                f"Created:\n{OUTPUT_VIDEO_NAME}\n{OUTPUT_CSV_NAME}"))

        except ProcessingStopped:
            self._log_message("\n--- Stopped by user ---")
            self._set_status("Stopped.")
        except Exception as e:
            self._log_message(f"An unexpected error occurred: {e}")
            self.root.after(0, lambda e=e: messagebox.showerror("Fatal Error", f"An unexpected error occurred:\n{e}"))
        finally:
            self.root.after(0, self._set_idle_state)

    def _prescan_clips(self, input_dir, depth_folder, video_files):
        """Verify every RGB/depth pair opens and shares one frame rate."""
        clips, problems = [], []
        fps_ref = None
        for filename in video_files:
            input_path = os.path.join(input_dir, filename)
            base_name, ext = os.path.splitext(filename)
            depth_path = os.path.join(depth_folder, f"{base_name}_depth{ext}")
            if not os.path.exists(depth_path):
                problems.append(f"Missing depth video for '{filename}' "
                                f"(expected '{os.path.basename(depth_path)}').")
                continue

            rgb_cap = cv2.VideoCapture(input_path)
            depth_cap = cv2.VideoCapture(depth_path)
            try:
                if not rgb_cap.isOpened():
                    problems.append(f"Cannot open '{filename}'.")
                    continue
                if not depth_cap.isOpened():
                    problems.append(f"Cannot open depth video for '{filename}'.")
                    continue
                _, _, fps, frames_rgb = self._get_video_properties(rgb_cap)
                _, _, _, frames_depth = self._get_video_properties(depth_cap)
                if frames_rgb <= 0 or fps <= 0:
                    problems.append(f"'{filename}' reports invalid frame count or fps.")
                    continue
                if frames_rgb != frames_depth:
                    self._log_message(
                        f"  [!] Warning: '{filename}' has {frames_rgb} RGB frames but "
                        f"{frames_depth} depth frames. Using the smaller count; check this "
                        f"pair if the control track drifts out of sync.")
                if fps_ref is None:
                    fps_ref = fps
                elif abs(fps - fps_ref) > 0.001:
                    problems.append(f"'{filename}' is {fps:.3f} fps but earlier clips are "
                                    f"{fps_ref:.3f} fps. All clips must share one frame rate.")
                    continue
                clips.append({
                    'filename': filename,
                    'rgb_path': input_path,
                    'depth_path': depth_path,
                    'fps': fps,
                    'frame_count': min(frames_rgb, frames_depth),
                })
            finally:
                rgb_cap.release()
                depth_cap.release()
        return clips, problems

    @staticmethod
    def _apply_box_filter(values, window):
        """Centered box filter with edge padding; window is clamped to clip length."""
        if window <= 1 or len(values) < 2:
            return list(values)
        effective_window = min(window, len(values))
        if effective_window % 2 == 0:
            effective_window = max(effective_window - 1, 1)
        if effective_window <= 1:
            return list(values)
        arr = np.asarray(values, dtype=float)
        pad = effective_window // 2
        padded = np.pad(arr, (pad, pad), mode='edge')
        kernel = np.ones(effective_window) / float(effective_window)
        return np.convolve(padded, kernel, mode='valid').tolist()

    def _get_video_properties(self, cap):
        """Helper to extract common video properties."""
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return width, height, fps, total_frames

    def _process_predict(self, rgb_path, depth_path, filename, params,
                         initial_prev_val=None, max_frames=None):
        """
        Analyze one RGB+depth pair and return (ema_values, last_prev_val),
        where ema_values is one float in 0-1 per decoded frame, or None on error.
        """
        self._set_status(f"Analyzing: {filename}")

        rgb_cap = cv2.VideoCapture(rgb_path)
        depth_cap = cv2.VideoCapture(depth_path)
        try:
            if not rgb_cap.isOpened() or not depth_cap.isOpened():
                self._log_message(f"  [!] Error: Could not open video files for {filename}. Skipping.")
                return None

            _, _, fps, total_frames_rgb = self._get_video_properties(rgb_cap)
            _, _, _, total_frames_depth = self._get_video_properties(depth_cap)

            total_frames = min(total_frames_rgb, total_frames_depth)
            if max_frames is not None:
                total_frames = min(total_frames, max_frames)
            if total_frames <= 0 or fps <= 0:
                self._log_message(f"  [!] Error: Invalid video properties. Skipping.")
                return None

            if self.estimator is None or getattr(self.estimator, 'model', None) is None:
                self._log_message("  > Loading Convergence Estimator...")
                try:
                    self.estimator = ConvergenceEstimator()
                except Exception as e:
                    self._log_message(f"  [!] Failed to load model: {e}")
                    return None
            estimator = self.estimator
            if getattr(estimator, 'model', None) is None:
                self._log_message("  [!] Error: Convergence model failed to initialize properly.")
                return None
            device = estimator.device

            scaler = EMAMinMaxScaler(decay=params['scaler_decay'], buffer_size=params['scaler_buffer'])
            rgb_queue = []
            pending_rgb, pending_depth = [], []
            raw_preds = []

            def run_pending_batch():
                """Run inference on the accumulated frame batch."""
                if not pending_rgb:
                    return
                try:
                    rgb_batch = torch.cat(pending_rgb, dim=0)
                    depth_batch = torch.cat(pending_depth, dim=0)
                    preds = estimator.predict(rgb_batch, depth_batch,
                                              user_ratio=params['convergence_ratio'])
                    raw_preds.extend(float(p) for p in preds)
                except Exception as e:
                    self._log_message(f"  [!] Inference error (batch of {len(pending_rgb)}): {e}")
                    fallback = raw_preds[-1] if raw_preds else 0.5
                    raw_preds.extend([fallback] * len(pending_rgb))
                pending_rgb.clear()
                pending_depth.clear()

            frames_read = 0
            for i in range(total_frames):
                if self.stop_event.is_set():
                    raise ProcessingStopped()

                ret_rgb, frame_rgb = rgb_cap.read()
                ret_depth, frame_depth = depth_cap.read()
                if not ret_rgb or not ret_depth:
                    break
                frames_read += 1

                # RGB: BGR uint8 -> [1, 3, h, w] float 0-1, downscaled to the
                # model input size immediately so the sync queue holds small
                # tensors instead of full-resolution frames.
                bgr_tensor = torch.from_numpy(frame_rgb).to(device)
                rgb_tensor = bgr_tensor[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255.0
                rgb_small = F.interpolate(rgb_tensor, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                                          mode="bilinear", align_corners=False)

                depth_bgr = torch.from_numpy(frame_depth).to(device)
                if depth_bgr.dim() == 3 and depth_bgr.shape[2] == 3:
                    # BGR to Grayscale using OpenCV coefficients
                    depth_tensor = (depth_bgr[:, :, 0] * 0.114 + depth_bgr[:, :, 1] * 0.587 + depth_bgr[:, :, 2] * 0.299)
                    depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0).float() / 255.0
                else:
                    depth_tensor = depth_bgr.unsqueeze(0).unsqueeze(0).float() / 255.0

                # Min/max is measured at full resolution (exact statistics) but
                # only the downscaled frame is buffered inside the scaler.
                frame_minmax = (depth_tensor.amin(), depth_tensor.amax())
                depth_small = F.interpolate(depth_tensor, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                                            mode="bilinear", align_corners=False)

                rgb_queue.append(rgb_small)
                scaled_depth = scaler.update(depth_small, minmax=frame_minmax)
                if scaled_depth is not None:
                    pending_rgb.append(rgb_queue.pop(0))
                    pending_depth.append(scaled_depth)
                    if len(pending_rgb) >= INFERENCE_BATCH_SIZE:
                        run_pending_batch()

                if i > 0 and i % 30 == 0:
                    self._set_status(f"Analyzing: {filename} ({i}/{total_frames})")

            # Flush remaining frames from the scaler buffer
            for scaled_depth in scaler.flush():
                if self.stop_event.is_set():
                    raise ProcessingStopped()
                pending_rgb.append(rgb_queue.pop(0))
                pending_depth.append(scaled_depth)
                if len(pending_rgb) >= INFERENCE_BATCH_SIZE:
                    run_pending_batch()
            run_pending_batch()

            if frames_read < total_frames:
                self._log_message(
                    f"  [!] Warning: only {frames_read} of {total_frames} frames could be "
                    f"decoded from {filename}. The control track uses the decoded count.")

            if not raw_preds:
                self._log_message("  [!] Error: No frames processed successfully.")
                return None

            # EMA smoothing over the raw per-frame predictions
            alpha = params['ema_alpha']
            values = []
            prev_val = initial_prev_val
            for pred in raw_preds:
                prev_val = pred if prev_val is None else alpha * pred + (1.0 - alpha) * prev_val
                values.append(prev_val)
            return values, prev_val
        finally:
            rgb_cap.release()
            depth_cap.release()

    # ------------------------------------------------------------------
    # Output writing
    # ------------------------------------------------------------------
    @staticmethod
    def _drain_pipe(pipe, sink):
        """Continuously read a pipe so FFmpeg can never block on a full stderr buffer."""
        try:
            for line in iter(pipe.readline, b''):
                sink.append(line.decode(errors='replace'))
        finally:
            pipe.close()

    def _write_output(self, output_dir, fps_str, clip_brightness, total_clips):
        """Stream all brightness values into one control video. Returns True on success."""
        final_output_path = os.path.join(output_dir, OUTPUT_VIDEO_NAME)
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'gray',
            '-s', f'{CONTROL_SIZE}x{CONTROL_SIZE}',
            '-r', fps_str,
            '-i', 'pipe:0',
            # Keep full-range values through the gray -> yuv420p conversion and
            # flag the stream as full range, so 0/255 survive to the Probe.
            '-vf', 'scale=in_range=pc:out_range=pc',
            '-color_range', 'pc',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '0', '-preset', 'ultrafast',
            '-vsync', 'cfr',
            final_output_path
        ]
        try:
            proc = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            )
        except FileNotFoundError:
            self._log_message("  [!] Error: FFmpeg not found. Please install FFmpeg and add it to PATH.")
            self.root.after(0, lambda: messagebox.showerror("Fatal Error", "FFmpeg not found on PATH."))
            return False

        stderr_lines = []
        drain_thread = threading.Thread(target=self._drain_pipe, args=(proc.stderr, stderr_lines), daemon=True)
        drain_thread.start()

        n_pixels = CONTROL_SIZE * CONTROL_SIZE
        frame_cache = {}
        stopped = False
        pipe_error = None
        try:
            for k, (clip, brightness_values) in enumerate(clip_brightness):
                self._set_status(f"Writing frames for: {clip['filename']}")
                for brightness in brightness_values:
                    if self.stop_event.is_set():
                        stopped = True
                        break
                    data = frame_cache.get(brightness)
                    if data is None:
                        data = bytes([brightness]) * n_pixels
                        frame_cache[brightness] = data
                    proc.stdin.write(data)
                if stopped:
                    break
                self._set_progress(total_clips + k + 1)
        except (BrokenPipeError, OSError) as e:
            pipe_error = e
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            drain_thread.join(timeout=5)

        if stopped:
            try:
                os.remove(final_output_path)
            except OSError:
                pass
            raise ProcessingStopped()

        stderr_text = ''.join(stderr_lines).strip()
        if pipe_error is not None or proc.returncode != 0:
            detail = stderr_text or str(pipe_error) or f"exit code {proc.returncode}"
            self._log_message(f"  [!] FFmpeg error: {detail}")
            return False
        if stderr_text:
            self._log_message(f"  [i] FFmpeg messages: {stderr_text}")
        self._log_message(f"\n  > Successfully created: {OUTPUT_VIDEO_NAME}")
        return True

    def _write_csv(self, output_dir, clip_values, clip_brightness):
        """Export the final convergence curve for diagnostics / alternate workflows."""
        csv_path = os.path.join(output_dir, OUTPUT_CSV_NAME)
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["global_frame", "clip", "clip_frame", "value", "brightness"])
                global_frame = 0
                for (clip, values), (_, brightness_values) in zip(clip_values, clip_brightness):
                    for clip_frame, (value, brightness) in enumerate(zip(values, brightness_values)):
                        writer.writerow([global_frame, clip['filename'], clip_frame,
                                         f"{value:.6f}", brightness])
                        global_frame += 1
            self._log_message(f"  > Exported curve data: {OUTPUT_CSV_NAME}")
        except OSError as e:
            self._log_message(f"  [!] Could not write CSV: {e}")


if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except (ImportError, AttributeError):
        pass

    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()
