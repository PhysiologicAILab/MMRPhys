'''
Remote Vital Signs Detection from Video
This script demonstrates how to estimate vital signs (Heart Rate and Respiration Rate) from a video file.
The script uses a pre-trained model to estimate vital signs from facial video data.

Usage:
    python infer_from_video.py --config config.yaml
'''

import torch
import onnxruntime as ort
import cv2
from pathlib import Path
import numpy as np
from scipy import signal
import json
from collections import OrderedDict
from PIL import Image, ImageTk
import collections
import argparse
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch
import time
from dataset.data_loader.face_detector.YOLO5Face import YOLO5Face
import yaml
import threading
from queue import Queue
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk


class SignalProcessor:
    def __init__(self, fs):
        self.fs = fs
        self.init_filters()

    def init_filters(self):
        lowcut_bvp = 0.6
        highcut_bvp = 3.3
        lowcut_rsp = 0.1
        highcut_rsp = 0.5
        order = 2
        nyquist = 0.5 * self.fs
        low_bvp = lowcut_bvp / nyquist
        high_bvp = highcut_bvp / nyquist
        low_rsp = lowcut_rsp / nyquist
        high_rsp = highcut_rsp / nyquist
        self.b_bvp, self.a_bvp = signal.butter(
            order, [low_bvp, high_bvp], btype='band')
        self.b_rsp, self.a_rsp = signal.butter(
            order, [low_rsp, high_rsp], btype='band')

    def next_power_of_2(self, x):
        """Calculate the nearest power of 2."""
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def calculate_fft_rate(self, phys_signal, low_pass=0.6, high_pass=3.3):
        """Calculate heart rate/ resp rate using Fast Fourier transform (FFT)."""
        phys_signal = np.expand_dims(phys_signal, 0)
        N = phys_signal.shape[1]
        if N <= 30*self.fs:
            nfft = self.next_power_of_2(N)
            f_phys, pxx_phys = periodogram(
                phys_signal, fs=self.fs, nfft=nfft, detrend=False)
        else:
            f_phys, pxx_phys = welch(phys_signal, fs=self.fs, nperseg=N//2, detrend=False)
        fmask_phys = np.argwhere((f_phys >= low_pass) & (f_phys <= high_pass))
        mask_phys = np.take(f_phys, fmask_phys)
        mask_pxx = np.take(pxx_phys, fmask_phys)
        fft_hr = np.take(mask_phys, np.argmax(mask_pxx, 0))[0] * 60
        return fft_hr

    def post_process(self, bvp, rsp):
        # apply normalization and bandpass filter
        bvp = (bvp - np.mean(bvp))/np.std(bvp)
        rsp = (rsp - np.mean(rsp))/np.std(rsp)
        bvp = signal.filtfilt(self.b_bvp, self.a_bvp, bvp)
        rsp = signal.filtfilt(self.b_rsp, self.a_rsp, rsp)
        return bvp, rsp

    def compute_metrics(self, bvp, rsp):
        # compute heart rate
        hr = self.calculate_fft_rate(bvp, low_pass=0.6, high_pass=3.3)
        # validate heart rate
        if hr < 40 or hr > 200:
            hr = np.nan
        else:
            hr = np.round(hr, 0)
        print(f'Heart rate: {hr}')

        # Compute respiration rate
        rr = self.calculate_fft_rate(rsp, low_pass=0.1, high_pass=0.5)
        # validate respiration rate
        if rr < 5 or rr > 40:
            rr = np.nan
        else:
            rr = np.round(rr, 0)
        print(f'Resp rate: {rr}')
        return hr, rr


class InferenceWorker:
    def __init__(self, config):
        self.model_path = Path(config['model']['path'])
        self.model_type = config['model']['type']
        self.num_frames = config['model']['input_shape']['num_frames']
        self.in_channels = config['model']['input_shape']['channels']
        self.height = config['model']['input_shape']['height']
        self.width = config['model']['input_shape']['width']
        self.fs = config['video']['sampling_rate']

        assert self.height == self.width, "Height and width must be equal"

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.init_model()


    def init_model(self):
        self.md_config = {}
        self.md_config["FRAME_NUM"] = self.num_frames
        self.md_config["TASKS"] = ["BVP", "RSP"]

        if self.model_type == 'torch':
            self.init_torch_model()
        elif self.model_type == 'onnx':
            self.init_onnx_model()
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def init_onnx_model(self):
        from tools.torch2onnx.MMRPhysSEF import MMRPhysSEF as rPhysModel

        self.model = rPhysModel(self.num_frames, self.md_config, self.in_channels)
        self.model = ort.InferenceSession(self.model_path)


    def init_torch_model(self):
        if self.height == 9:
            from neural_methods.model.MMRPhys.MMRPhysSEF import MMRPhysSEF as rPhysModel
        elif self.height == 36:
            from neural_methods.model.MMRPhys.MMRPhysMEF import MMRPhysMEF as rPhysModel
        elif self.height == 72:
            from neural_methods.model.MMRPhys.MMRPhysLEF import MMRPhysLEF as rPhysModel
        else:
            raise ValueError(f"Invalid height and width: {self.height, self.width}")

        self.model = rPhysModel(self.num_frames, self.md_config, self.in_channels)
        model_weights = torch.load(self.model_path, map_location=self.device)
        model_weights = self.map_weights(model_weights)

        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(model_weights)
        self.model.to(self.device)
        self.model.eval()


    def map_weights(self, old_state_dict):
        """Maps weights from old Sequential ConvBlock3D to new implementation"""
        new_state_dict = OrderedDict()

        for key, value in old_state_dict.items():
            # Skip unnecessary weights
            if any(skip in key for skip in ['fsam', 'bias1']):
                continue
            else:
                new_state_dict[key] = value

        return new_state_dict


    def infer_rphys(self, frame_buffer):
        if self.model_type == 'onnx':
            output = self.model.run(None, {'input': frame_buffer.astype(np.float32)})
            bvp = output[0]
            rsp = output[1]

        else:
            with torch.no_grad():
                torch_frames = torch.from_numpy(frame_buffer).float().to(self.device)
                output = self.model(torch_frames)
                bvp = output[0].cpu().numpy()
                rsp = output[1].cpu().numpy()
        return bvp, rsp


class videoProcessor:
    def __init__(self, video_file):
        self.video_file = video_file
        self.cap = None
        self.is_live_capture = isinstance(video_file, int)
        self.initialize_video_stream()
        if self.is_live_capture:
            self.max_frames = float('inf')  # For live capture, set to infinite
        else:
            self.max_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def initialize_video_stream(self):
        try:
            print(
                f"Attempting to open {'camera' if isinstance(self.video_file, int) else 'video file'}: {self.video_file}")
            self.cap = cv2.VideoCapture(self.video_file)

            if not self.cap.isOpened():
                raise IOError("Failed to open video capture")

            # Print camera properties for live capture
            if isinstance(self.video_file, int):
                width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                print(
                    f"Camera initialized: Resolution {width}x{height}, FPS: {fps}")

                # Try setting camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

                # Verify settings
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                print(
                    f"Camera settings after initialization: Resolution {actual_width}x{actual_height}, FPS: {actual_fps}")

        except Exception as e:
            print(f"Error initializing video capture: {str(e)}")
            raise

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            print("Error: Video capture is not initialized or has been closed")
            return None

        try:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_live_capture:
                    print("Error reading frame from camera")
                return None

            if frame is None:
                print("Received null frame")
                return None

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame

        except Exception as e:
            print(f"Error reading frame: {str(e)}")
            return None

    def __del__(self):
        if self.cap is not None:
            print("Releasing video capture")
            self.cap.release()

class FaceDetector:
    def __init__(self):
        self.faceDetector = YOLO5Face()
        self.reset()
        self.face = None
        self.count_no_face = 0
    
    def reset(self):
        self.face_detected = False

    def detect_face(self, frame):
        if not self.face_detected:
            face = self.faceDetector.detect_face(frame)
            if face is not None:
                print(f"Face detected! Bbox: {face}")  # Debug print
                self.face_detected = True
                self.face = face
                self.count_no_face = 0
            else:
                self.count_no_face += 1
                self.face_detected = False
                if self.count_no_face % 30 == 0:  # Print every 30 frames
                    print(f"No face detected for {self.count_no_face} frames")

        if self.count_no_face > 10:
            self.reset()
            return None

        return self.face


class SignalPlotter:
    def __init__(self, fs, plot_duration, is_live_capture):
        self.fs = fs
        self.plot_duration = plot_duration
        self.is_live_capture = is_live_capture
        self.setup_plot()
        self.last_update = time.time()
        self.update_interval = 1/30 if is_live_capture else 1/60  # Faster updates for recorded video
        self.should_stop = False
        self.is_paused = False

    def setup_plot(self):
        self.root = tk.Tk()
        self.root.title("Remote Vital Signs Monitor")

        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create video frame with fixed size for face ROI
        self.video_frame = tk.Frame(self.main_frame)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Create video label with fixed size
        # Fixed size for face ROI
        self.video_label = tk.Label(self.video_frame, width=360, height=360)
        self.video_label.pack()

        # Add ROI status indicator
        self.roi_status = tk.Label(self.video_frame, text="Face ROI Status: Detecting...",
                                   fg="orange")
        self.roi_status.pack()

        # Create plots frame
        self.plots_frame = tk.Frame(self.main_frame)
        self.plots_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Setup matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plots_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Setup initial plot lines
        plot_samples = int(self.plot_duration * self.fs)
        self.time_axis = np.linspace(0, self.plot_duration, plot_samples)
        zeros = np.zeros(plot_samples)

        self.ax1.set_title('BVP Signal | HR: -- bpm')
        self.ax2.set_title('RSP Signal | RR: -- br/min')
        self.line_bvp, = self.ax1.plot(self.time_axis, zeros)
        self.line_rsp, = self.ax2.plot(self.time_axis, zeros)

        # Add controls
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack(fill=tk.X)

        self.stop_button = tk.Button(
            self.controls_frame, text="Stop", command=self.stop_processing)
        self.stop_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.pause_button = tk.Button(
            self.controls_frame, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.is_paused = False
        self.should_stop = False

    def stop_processing(self):
        self.should_stop = True

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.pause_button.config(text="Resume" if self.is_paused else "Pause")

    def update_plot(self, face_frame, bvp_data, rsp_data, hr, rr, face_detected):
        if not hasattr(self, 'root') or self.is_paused:
            return

        # For recorded video, update as fast as possible while maintaining readable display
        if self.is_live_capture:
            current_time = time.time()
            if (current_time - self.last_update) < self.update_interval:
                return
            self.last_update = current_time
        else:
            # For recorded video, just add a tiny sleep to prevent GUI from freezing
            time.sleep(0.001)

        # Update face ROI frame
        if face_frame is not None and face_detected:
            # Resize face ROI to fixed display size while maintaining aspect ratio
            display_size = (360, 360)
            face_h, face_w = face_frame.shape[:2]
            scale = min(display_size[0]/face_w, display_size[1]/face_h)
            new_size = (int(face_w * scale), int(face_h * scale))

            face_frame = cv2.resize(face_frame, new_size)

            # Create black canvas of fixed size
            canvas = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)

            # Center the face ROI in the canvas
            y_offset = (display_size[1] - new_size[1]) // 2
            x_offset = (display_size[0] - new_size[0]) // 2
            canvas[y_offset:y_offset+new_size[1],
                   x_offset:x_offset+new_size[0]] = face_frame

            # Convert to PIL Image - face_frame is already in RGB, so no need to convert
            img = Image.fromarray(canvas)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Update ROI status
            self.roi_status.config(text="Face ROI Status: Detected", fg="green")
        else:
            # Display blank frame when no face is detected
            canvas = np.zeros((360, 360, 3), dtype=np.uint8)
            img = Image.fromarray(canvas)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.roi_status.config(text="Face ROI Status: No Face Detected", fg="red")

        # Smooth update of signal plots
        plot_samples = int(self.plot_duration * self.fs)
        if len(bvp_data) > plot_samples:
            bvp_data = bvp_data[-plot_samples:]
            rsp_data = rsp_data[-plot_samples:]

        # Pad with zeros if needed
        if len(bvp_data) < plot_samples:
            bvp_data = np.pad(bvp_data, (plot_samples - len(bvp_data), 0))
            rsp_data = np.pad(rsp_data, (plot_samples - len(rsp_data), 0))

        self.line_bvp.set_ydata(bvp_data)
        self.line_rsp.set_ydata(rsp_data)

        # Update titles with metrics only when face is detected
        if face_detected:
            self.ax1.set_title(f'BVP Signal | HR: {hr:.0f} bpm')
            self.ax2.set_title(f'RSP Signal | RR: {rr:.0f} br/min')
        else:
            self.ax1.set_title('BVP Signal | HR: -- bpm')
            self.ax2.set_title('RSP Signal | RR: -- br/min')

        # Update axis limits for better visualization
        self.ax1.set_ylim(np.min(bvp_data)-0.1, np.max(bvp_data)+0.1)
        self.ax2.set_ylim(np.min(rsp_data)-0.1, np.max(rsp_data)+0.1)

        self.canvas.draw()
        self.root.update()

    def stop_processing(self):
        self.should_stop = True
        
    def cleanup(self):
        """Safe cleanup method to be called from main thread"""
        try:
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.quit()
                self.root.destroy()
        except Exception:
            # If the root is already destroyed, just pass
            pass


class RemoteVitalSigns:
    def __init__(self, config, config_path):
        self.model_path = Path(config['model']['path'])
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            self.video_file = int(config['video']['path'])
        except:
            self.video_file = Path(config['video']['path'])
            if not self.video_file.exists():
                raise FileNotFoundError(f"Video file not found: {self.video_file}")

        # Initialize all parameters from config
        self.model_type = config['model']['type']
        self.num_frames = config['model']['input_shape']['num_frames']
        self.in_channels = config['model']['input_shape']['channels']
        self.height = config['model']['input_shape']['height']
        self.width = config['model']['input_shape']['width']
        self.fs = config['video']['sampling_rate']
        self.face_detection_interval = int(config['processing']['face_detection_interval'])
        self.inference_interval = int(config['processing']['inference_interval'])
        self.plot_duration = config['processing']['plot_duration']

        # Initialize components
        self.signal_processor = SignalProcessor(self.fs)
        self.video_processor = videoProcessor(self.video_file)
        self.max_frames = self.video_processor.max_frames
        self.model_instance = InferenceWorker(config)
        self.face_detector = FaceDetector()
        
        # Initialize buffers
        self.init_buffers()
        
        # Initialize threading components
        self.frame_queue = Queue(maxsize=100)
        self.result_queue = Queue()
        self.stop_event = threading.Event()

        # Determine if we're using live capture or recorded video
        self.is_live_capture = isinstance(self.video_file, int)

        # Initialize plotter
        self.plotter = SignalPlotter(self.fs, self.plot_duration, self.is_live_capture)
        
        # Additional buffer initialization for smooth updates
        self.signal_buffer_size = int(self.fs * self.plot_duration)
        self.bvp_buffer = collections.deque(maxlen=self.signal_buffer_size)
        self.rsp_buffer = collections.deque(maxlen=self.signal_buffer_size)

        # Setup output files
        config_stem = Path(config_path).stem
        # Extract video name, append timestamp for live video
        if isinstance(self.video_file, Path):
            root_path = self.video_file.parent
            vid_name = self.video_file.stem
        else:
            # Default to Downloads folder for live capture
            root_path = Path.home().joinpath('Downloads', 'rPhys')
            vid_name = f"{time.strftime('%Y%m%d_%H%M%S')}"
            root_path.mkdir(exist_ok=True, parents=True)

        self.json_filename = root_path.joinpath(f"{config_stem}_{vid_name}_estimated.json")
        self.plot_filename = root_path.joinpath(f"{config_stem}_{vid_name}_estimated.png")


    def init_buffers(self):
        self.inference_frame_buffer = np.zeros(
            (1, self.in_channels, self.num_frames, self.height, self.width))
        self.estimated_bvp = np.array([])
        self.estimated_rsp = np.array([])
        self.hr_list = []
        self.rr_list = []

    def video_capture_thread(self):
        count_frame = 0
        face = None
        last_frame_time = time.time()
        frames_processed = 0
        frames_with_face = 0

        print("Starting video capture thread...")

        # Add initial frame check
        initial_frame = self.video_processor.get_frame()
        if initial_frame is None:
            print("Failed to get initial frame - check camera permissions and connection")
            self.frame_queue.put(None)
            return

        print(f"Successfully got initial frame with shape: {initial_frame.shape}")

        while not self.stop_event.is_set() and count_frame < self.max_frames:
            if self.plotter.is_paused:
                time.sleep(0.01)
                continue

            frame = self.video_processor.get_frame()
            if frame is None:
                print("Failed to get frame - camera disconnected or stream ended")
                break

            frames_processed += 1
            if count_frame % self.face_detection_interval == 0:
                face = self.face_detector.detect_face(frame)

            if face is not None:
                frames_with_face += 1
                x1, y1, x2, y2 = face
                face_frame = frame[y1:y2, x1:x2]
                processed_frame = cv2.resize(face_frame, (self.width, self.height))
                processed_frame = processed_frame[np.newaxis, :, :, :]
                processed_frame = processed_frame.transpose(0, 3, 1, 2)
                
                try:
                    self.frame_queue.put((count_frame, face_frame, processed_frame, True), 
                                    timeout=0.01)
                except Queue.Full:
                    print("Frame queue full!")  # Debug print
                    continue
            else:
                try:
                    self.frame_queue.put((count_frame, None, None, False), 
                                    timeout=0.01)
                except Queue.Full:
                    continue

            count_frame += 1

        print(f"Video capture thread ended. Processed {frames_processed} frames total, {frames_with_face} with face")  # Debug print
        self.frame_queue.put(None)

    def inference_thread(self):
        count_frame = 0
        measurements_recorded = 0

        print("Starting inference thread...")  # Debug print

        while not self.stop_event.is_set():
            data = self.frame_queue.get()
            if data is None:
                break

            if self.plotter.is_paused:
                continue

            frame_idx, face_frame, processed_frame, face_detected = data

            if face_detected and processed_frame is not None:
                if count_frame < self.num_frames:
                    self.inference_frame_buffer[:, :, count_frame, :, :] = processed_frame
                else:
                    self.inference_frame_buffer[:, :, :-1, :, :] = self.inference_frame_buffer[:, :, 1:, :, :]
                    self.inference_frame_buffer[:, :, -1, :, :] = processed_frame

                if count_frame >= self.num_frames and count_frame % self.inference_interval == 0:
                    bvp, rsp = self.model_instance.infer_rphys(self.inference_frame_buffer)
                    bvp, rsp = self.signal_processor.post_process(bvp, rsp)

                    self.bvp_buffer.extend(bvp.reshape(-1))
                    self.rsp_buffer.extend(rsp.reshape(-1))

                    hr, rr = self.signal_processor.compute_metrics(
                        np.array(list(self.bvp_buffer)),
                        np.array(list(self.rsp_buffer))
                    )

                    if not np.isnan(hr) and not np.isnan(rr):
                        measurements_recorded += 1
                        if measurements_recorded % 10 == 0:  # Print every 10 measurements
                            print(f"Recorded {measurements_recorded} valid measurements. HR: {hr:.1f}, RR: {rr:.1f}")

                    self.result_queue.put((face_frame, np.array(list(self.bvp_buffer)),
                                        np.array(list(self.rsp_buffer)), hr, rr, face_detected))
            else:
                self.result_queue.put((None, np.array(list(self.bvp_buffer)) if self.bvp_buffer else np.zeros(self.signal_buffer_size),
                                    np.array(list(self.rsp_buffer)) if self.rsp_buffer else np.zeros(self.signal_buffer_size),
                                    np.nan, np.nan, False))

            count_frame += 1

        print(f"Inference thread ended. Recorded {measurements_recorded} valid measurements")  # Debug print
        self.result_queue.put(None)

    def plotting_thread(self):
        saved_metrics = {
            'timestamps': [],
            'hr_values': [],
            'rr_values': [],
            'bvp_signal': [],
            'rsp_signal': []
        }

        start_time = time.time()

        try:
            while not self.stop_event.is_set():
                if self.plotter.should_stop:
                    self.stop_event.set()
                    break

                data = self.result_queue.get()
                if data is None:
                    break

                face_frame, bvp, rsp, hr, rr, face_detected = data

                # Update the plot
                self.plotter.update_plot(
                    face_frame, bvp, rsp, hr, rr, face_detected)

                # Save metrics only when face is detected
                if face_detected and not np.isnan(hr) and not np.isnan(rr):
                    current_time = time.time() - start_time
                    saved_metrics['timestamps'].append(current_time)
                    saved_metrics['hr_values'].append(hr)
                    saved_metrics['rr_values'].append(rr)

                    if len(bvp) > 0:
                        saved_metrics['bvp_signal'].extend(
                            bvp[-self.inference_interval:])
                    if len(rsp) > 0:
                        saved_metrics['rsp_signal'].extend(
                            rsp[-self.inference_interval:])

            # Save final results
            if len(saved_metrics['hr_values']) > 0:
                # Calculate average metrics
                avg_hr = np.nanmean(saved_metrics['hr_values'])
                avg_rr = np.nanmean(saved_metrics['rr_values'])

                # Save metrics to JSON
                results = {
                    'average_hr': float(avg_hr),
                    'average_rr': float(avg_rr),
                    'hr_values': saved_metrics['hr_values'],
                    'rr_values': saved_metrics['rr_values'],
                    'timestamps': saved_metrics['timestamps'],
                    'sampling_rate': self.fs,
                    'bvp_signal': saved_metrics['bvp_signal'],
                    'rsp_signal': saved_metrics['rsp_signal'],
                    'duration': saved_metrics['timestamps'][-1] if saved_metrics['timestamps'] else 0
                }

                with open(self.json_filename, 'w') as f:
                    json.dump(results, f, indent=4)

                # Create and save final plots
                plt.figure(figsize=(12, 8))

                # Plot HR over time
                plt.subplot(3, 1, 1)
                plt.plot(saved_metrics['timestamps'],
                         saved_metrics['hr_values'], 'b-')
                plt.title(f'Heart Rate Over Time (Average: {avg_hr:.1f} bpm)')
                plt.xlabel('Time (s)')
                plt.ylabel('Heart Rate (bpm)')
                plt.grid(True)

                # Plot RR over time
                plt.subplot(3, 1, 2)
                plt.plot(saved_metrics['timestamps'],
                         saved_metrics['rr_values'], 'g-')
                plt.title(
                    f'Respiration Rate Over Time (Average: {avg_rr:.1f} br/min)')
                plt.xlabel('Time (s)')
                plt.ylabel('Resp. Rate (br/min)')
                plt.grid(True)

                # Plot final signal segments
                plt.subplot(3, 1, 3)
                time_axis = np.arange(
                    len(saved_metrics['bvp_signal'])) / self.fs
                plt.plot(
                    time_axis, saved_metrics['bvp_signal'], 'b-', label='BVP', alpha=0.7)
                plt.plot(
                    time_axis, saved_metrics['rsp_signal'], 'g-', label='RSP', alpha=0.7)
                plt.title('Final Signal Segments')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.grid(True)

                plt.tight_layout()
                plt.savefig(self.plot_filename)
                plt.close()

                print(f"\nProcessing complete!")
                print(f"Average Heart Rate: {avg_hr:.1f} bpm")
                print(f"Average Respiration Rate: {avg_rr:.1f} br/min")
                print(f"Results saved to: {self.json_filename}")
                print(f"Plots saved to: {self.plot_filename}")
            else:
                print("\nNo valid measurements were recorded.")

        except Exception as e:
            print(f"Error in plotting thread: {e}")
        finally:
            # Schedule GUI cleanup on the main thread
            if hasattr(self.plotter, 'root'):
                self.plotter.root.after(100, self.plotter.cleanup)


    def run_inference(self):
        # Start threads
        video_thread = threading.Thread(target=self.video_capture_thread)
        inference_thread = threading.Thread(target=self.inference_thread)
        plot_thread = threading.Thread(target=self.plotting_thread)

        video_thread.start()
        inference_thread.start()
        plot_thread.start()

        try:
            # Keep main thread alive for GUI
            self.plotter.root.mainloop()
        except KeyboardInterrupt:
            print("\nStopping processing...")
        finally:
            self.stop_event.set()
            self.plotter.should_stop = True

        # Wait for threads to complete
        video_thread.join()
        inference_thread.join()
        plot_thread.join()


    def save_estimated_signals(self):
        estimated_signals = {
            'hr_list': self.hr_list,
            'rr_list': self.rr_list,
            'bvp': self.estimated_bvp.tolist(),
            'rsp': self.estimated_rsp.tolist()
        }
        
        with open(self.json_filename, 'w') as f:
            json.dump(estimated_signals, f)

    def save_plots(self):
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.estimated_bvp)
        plt.title('Estimated BVP | Average HR: {:.0f}'.format(np.mean(np.array(self.hr_list))))
        plt.subplot(2, 1, 2)
        plt.plot(self.estimated_rsp)
        plt.title('Estimated RSP | Average RR: {:.0f}'.format(np.mean(np.array(self.rr_list))))
        plt.tight_layout()
        plt.savefig(self.plot_filename)
        plt.close()


    def workflow(self):
        try:
            self.run_inference()
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            # Ensure cleanup is done in the main thread
            try:
                if hasattr(self, 'plotter'):
                    self.plotter.cleanup()
            except Exception as e:
                print(f"Cleanup error (can be safely ignored): {e}")
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Remote Vital Signs Detection')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Check if using live capture
    try:
        video_path = config['video']['path']
        is_live = isinstance(video_path, int) or video_path.isdigit()
        if is_live:
            print(f"Using live capture with camera index: {video_path}")
            # Try opening camera briefly to check if it works
            cap = cv2.VideoCapture(int(video_path))
            if not cap.isOpened():
                print("Error: Could not open camera. Please check if:")
                print("1. Camera is properly connected")
                print("2. Camera permissions are granted")
                print("3. Camera is not being used by another application")
                return
            cap.release()
    except Exception as e:
        print(f"Error checking camera: {str(e)}")
        return

    rPhysObj = RemoteVitalSigns(config, args.config)
    rPhysObj.workflow()

if __name__ == '__main__':
    main()