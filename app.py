import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import cv2
import numpy as np
import threading
import torch
from convergence_estimator import ConvergenceEstimator
from depth_scaler import EMAMinMaxScaler

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
        self.root.geometry("700x650")
        self.root.minsize(600, 600)

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
        input_folder_label = ttk.Label(main_frame, textvariable=self.input_folder_var, wraplength=450, foreground="gray")
        input_folder_label.grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(main_frame, text="Browse...", command=self.select_input_folder).grid(row=1, column=2, sticky="e")

        ttk.Label(main_frame, text="Input (Depth) Folder:").grid(row=2, column=0, sticky="w")
        self.depth_folder_var = tk.StringVar(value="No folder selected")
        depth_folder_label = ttk.Label(main_frame, textvariable=self.depth_folder_var, wraplength=450, foreground="gray")
        depth_folder_label.grid(row=2, column=1, sticky="ew", padx=5)
        self.depth_browse_btn = ttk.Button(main_frame, text="Browse...", command=self.select_depth_folder)
        self.depth_browse_btn.grid(row=2, column=2, sticky="e")

        ttk.Label(main_frame, text="Output Folder:").grid(row=3, column=0, sticky="w")
        self.output_folder_var = tk.StringVar(value="No folder selected")
        output_folder_label = ttk.Label(main_frame, textvariable=self.output_folder_var, wraplength=450, foreground="gray")
        output_folder_label.grid(row=3, column=1, sticky="ew", padx=5)
        ttk.Button(main_frame, text="Browse...", command=self.select_output_folder).grid(row=3, column=2, sticky="e")

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
        self.scaler_buffer_var = tk.StringVar(value="60")
        self.scaler_buffer_entry = ttk.Entry(self.convergence_frame, textvariable=self.scaler_buffer_var, width=10)
        self.scaler_buffer_entry.grid(row=1, column=3, sticky="w", padx=5, pady=(10, 0))

        ttk.Label(self.convergence_frame, text="Temporal Window (frames):").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.temporal_window_var = tk.StringVar(value="31")
        self.temporal_window_entry = ttk.Entry(self.convergence_frame, textvariable=self.temporal_window_var, width=10)
        self.temporal_window_entry.grid(row=2, column=1, sticky="w", padx=5, pady=(10, 0))

        # --- Progress Bar and Status ---
        self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal", mode="determinate")
        self.progress_bar.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(20, 5))

        self.status_label = ttk.Label(main_frame, text="Waiting to start...")
        self.status_label.grid(row=6, column=0, columnspan=3, sticky="ew", padx=5)

        # --- Log/Status Display ---
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding=10)
        log_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(10, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(7, weight=1)

        self.log_text = tk.Text(log_frame, height=10, state="disabled", wrap="word", bg="#f0f0f0", relief="sunken", borderwidth=1)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=scrollbar.set)

        # --- Action Button ---
        self.start_button = ttk.Button(main_frame, text="Start Processing", command=self.start_processing_thread)
        self.start_button.grid(row=8, column=0, columnspan=3, pady=(10, 0), sticky="ew")

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

    def select_input_folder(self):
        folder_path = filedialog.askdirectory(title="Select Input Folder")
        if folder_path:
            self.input_folder_var.set(folder_path)
            for child in self.root.winfo_children()[0].winfo_children():
                if isinstance(child, ttk.Label) and child.cget("textvariable") == str(self.input_folder_var):
                    child.config(foreground="black")
                    break

    def select_depth_folder(self):
        folder_path = filedialog.askdirectory(title="Select Depth Folder")
        if folder_path:
            self.depth_folder_var.set(folder_path)
            for child in self.root.winfo_children()[0].winfo_children():
                if isinstance(child, ttk.Label) and child.cget("textvariable") == str(self.depth_folder_var):
                    child.config(foreground="black")
                    break

    def select_output_folder(self):
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self.output_folder_var.set(folder_path)
            for child in self.root.winfo_children()[0].winfo_children():
                 if isinstance(child, ttk.Label) and child.cget("textvariable") == str(self.output_folder_var):
                    child.config(foreground="black")
                    break
    
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
            
        params = {}
        try:
            params['convergence_ratio'] = float(self.convergence_ratio_var.get())
            if not (0.0 <= params['convergence_ratio'] <= 1.0): raise ValueError
            params['ema_alpha'] = float(self.ema_alpha_var.get())
            if not (0.0 <= params['ema_alpha'] <= 1.0): raise ValueError
            params['scaler_decay'] = float(self.scaler_decay_var.get())
            if not (0.0 <= params['scaler_decay'] <= 1.0): raise ValueError
            params['scaler_buffer'] = int(self.scaler_buffer_var.get())
            if params['scaler_buffer'] < 1: raise ValueError
            params['temporal_window'] = int(self.temporal_window_var.get())
            if params['temporal_window'] < 1: raise ValueError
            if params['temporal_window'] % 2 == 0: 
                params['temporal_window'] += 1 # Enforce odd number for a perfectly centered window
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters. Please check ratio, alpha, decay (floats 0-1) and buffers (int >= 1).")
            return

        params['depth_folder'] = depth_folder

        self.start_button.config(state="disabled", text="Processing...")
        self._set_progress(0)
        
        processing_thread = threading.Thread(
            target=self.process_videos_controller,
            args=(input_folder, output_folder, params),
            daemon=True
        )
        processing_thread.start()

    def process_videos_controller(self, input_dir, output_dir, params):
        try:
            self._log_message("--- Starting Process ---")
            self._log_message("Mode: Convergence Model (predict)")
            video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
            if not video_files:
                self._log_message("No video files found in the input directory.")
                self.root.after(0, lambda: messagebox.showwarning("Warning", "No video files found."))
                return

            self.root.after(0, self.progress_bar.config, {'maximum': len(video_files)})
            
            final_output_path = os.path.join(output_dir, "M2SVid_Convergence_Control.mp4")
            out_writer = None
            
            for i, filename in enumerate(video_files):
                self._log_message(f"\nProcessing '{filename}' ({i+1}/{len(video_files)})...")
                input_path = os.path.join(input_dir, filename)

                base_name, ext = os.path.splitext(filename)
                depth_filename = f"{base_name}_depth{ext}"
                depth_path = os.path.join(params['depth_folder'], depth_filename)
                if not os.path.exists(depth_path):
                    self._log_message(f"  [!] Error: Depth video not found for {filename}. Skipping.")
                    continue
                    
                result = self._process_predict(
                    input_path, depth_path, filename, 
                    params['convergence_ratio'], params['ema_alpha'],
                    params['scaler_decay'], params['scaler_buffer'],
                    params['temporal_window']
                )
                
                if result is None:
                    continue
                    
                width, height, fps, total_frames, predicted_brightness_values = result
                
                if out_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_writer = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height), isColor=False)
                    if not out_writer.isOpened():
                        self._log_message(f"  [!] Error: Could not create output video writer.")
                        self.root.after(0, lambda: messagebox.showerror("Fatal Error", "Could not create output video writer."))
                        return
                
                self._set_status(f"Writing frames for: {filename}")
                for j in range(total_frames):
                    # Ensure brightness_values list has enough frames, otherwise use last known value
                    brightness = predicted_brightness_values[j] if j < len(predicted_brightness_values) else predicted_brightness_values[-1]
                    frame = np.full((height, width, 1), brightness, dtype=np.uint8)
                    out_writer.write(frame)
                
                self._set_progress(i + 1)

            if out_writer is not None:
                out_writer.release()
                self._log_message(f"\n  > Successfully created: {os.path.basename(final_output_path)}")

            self._log_message("\n--- Processing Complete ---")
            self._set_status("Finished.")
            self.root.after(0, lambda: messagebox.showinfo("Success", "All videos have been processed successfully!"))

        except Exception as e:
            self._log_message(f"An unexpected error occurred: {e}")
            self.root.after(0, lambda e=e: messagebox.showerror("Fatal Error", f"An unexpected error occurred:\n{e}"))
        
        finally:
            self.root.after(0, self.start_button.config, {'state': "normal", 'text': "Start Processing"})

    def _get_video_properties(self, cap):
        """Helper to extract common video properties."""
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return width, height, fps, total_frames

    def _process_predict(self, rgb_path, depth_path, filename, convergence_ratio, ema_alpha, scaler_decay, scaler_buffer, temporal_window):
        self._set_status(f"Analyzing: {filename}")
        
        rgb_cap = cv2.VideoCapture(rgb_path)
        depth_cap = cv2.VideoCapture(depth_path)
        
        if not rgb_cap.isOpened() or not depth_cap.isOpened():
            self._log_message(f"  [!] Error: Could not open video files. Skipping.")
            return None
            
        width, height, fps, total_frames_rgb = self._get_video_properties(rgb_cap)
        _, _, _, total_frames_depth = self._get_video_properties(depth_cap)
        
        total_frames = min(total_frames_rgb, total_frames_depth)
        if total_frames == 0 or fps == 0:
            self._log_message(f"  [!] Error: Invalid video properties. Skipping.")
            rgb_cap.release()
            depth_cap.release()
            return None
            
        if getattr(self, 'estimator', None) is None:
            self._log_message(f"  > Loading Convergence Estimator...")
            try:
                self.estimator = ConvergenceEstimator()
            except Exception as e:
                self._log_message(f"  [!] Failed to load model: {e}")
                rgb_cap.release()
                depth_cap.release()
                return None
                
        estimator = self.estimator
        device = estimator.device
            
        if estimator is None or getattr(estimator, 'model', None) is None:
            self._log_message("  [!] Error: Convergence model failed to initialize properly.")
            rgb_cap.release()
            depth_cap.release()
            return None

        predicted_brightness_values = []
        prev_val = None
        
        scaler = EMAMinMaxScaler(decay=scaler_decay, buffer_size=scaler_buffer)
        rgb_queue = []
        
        for i in range(total_frames):
            ret_rgb, frame_rgb = rgb_cap.read()
            ret_depth, frame_depth = depth_cap.read()
            
            if not ret_rgb or not ret_depth:
                break
                
            # Upload directly to GPU: Convert RGB to float tensor [1, 3, H, W] 
            bgr_tensor = torch.from_numpy(frame_rgb).to(device)
            rgb_tensor = bgr_tensor[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # Upload directly to GPU: Convert Depth to float tensor [1, 1, H, W]
            depth_bgr = torch.from_numpy(frame_depth).to(device)
            if depth_bgr.dim() == 3 and depth_bgr.shape[2] == 3:
                # BGR to Grayscale using OpenCV coefficients
                depth_tensor = (depth_bgr[:, :, 0] * 0.114 + depth_bgr[:, :, 1] * 0.587 + depth_bgr[:, :, 2] * 0.299)
                depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0).float() / 255.0
            else:
                depth_tensor = depth_bgr.unsqueeze(0).unsqueeze(0).float() / 255.0
            
            rgb_queue.append(rgb_tensor)
            scaled_depth = scaler.update(depth_tensor, return_minmax=False)
            
            if scaled_depth is not None:
                sync_rgb = rgb_queue.pop(0)
                # Predict
                try:
                    preds = estimator.predict(sync_rgb, scaled_depth, user_ratio=convergence_ratio)
                    pred_val = preds[0] # list of float
                    
                    if prev_val is None:
                        smoothed_val = pred_val
                    else:
                        smoothed_val = ema_alpha * pred_val + (1.0 - ema_alpha) * prev_val
                        
                    prev_val = smoothed_val
                    predicted_brightness_values.append(int(smoothed_val * 255))
                except Exception as e:
                    self._log_message(f"  [!] Inference error frame {i}: {e}")
                    fallback = 0.5 if prev_val is None else prev_val
                    predicted_brightness_values.append(int(fallback * 255))
                    prev_val = fallback
                
            if i > 0 and i % 30 == 0:
                self._set_status(f"Analyzing: {filename} ({i}/{total_frames})")
                
        # Flush remaining frames from buffer
        flushed_depths = scaler.flush(return_minmax=False)
        for scaled_depth in flushed_depths:
            sync_rgb = rgb_queue.pop(0)
            try:
                preds = estimator.predict(sync_rgb, scaled_depth, user_ratio=convergence_ratio)
                pred_val = preds[0]
                
                if prev_val is None:
                    smoothed_val = pred_val
                else:
                    smoothed_val = ema_alpha * pred_val + (1.0 - ema_alpha) * prev_val
                    
                prev_val = smoothed_val
                predicted_brightness_values.append(int(smoothed_val * 255))
            except Exception as e:
                self._log_message(f"  [!] Inference error (flush): {e}")
                fallback = 0.5 if prev_val is None else prev_val
                predicted_brightness_values.append(int(fallback * 255))
                prev_val = fallback

        rgb_cap.release()
        depth_cap.release()
        
        if not predicted_brightness_values:
            self._log_message("  [!] Error: No frames processed successfully.")
            return None
            
        if temporal_window > 1:
            # Clamp window to clip length so short clips still get smoothed
            effective_window = min(temporal_window, len(predicted_brightness_values))
            if effective_window % 2 == 0:
                effective_window = max(effective_window - 1, 1)  # Keep odd for symmetry
            if effective_window > 1:
                self._log_message(f"  > Applying Temporal Smoothing (Window={effective_window} of {temporal_window})...")
                arr = np.array(predicted_brightness_values, dtype=float)
                pad_size = effective_window // 2
                padded_arr = np.pad(arr, (pad_size, pad_size), mode='edge')
                box_filter = np.ones(effective_window) / float(effective_window)
                smoothed_arr = np.convolve(padded_arr, box_filter, mode='valid')
                predicted_brightness_values = np.clip(smoothed_arr, 0, 255).astype(int).tolist()
        
        return width, height, fps, total_frames, predicted_brightness_values


if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except (ImportError, AttributeError):
        pass 

    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()
