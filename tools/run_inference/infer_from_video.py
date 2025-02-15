import torch
import onnxruntime as ort
import cv2
from pathlib import Path
import numpy as np
from scipy import signal
import json
from collections import OrderedDict
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
        self.initialize_video_stream()
        self.max_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def initialize_video_stream(self):
        try:
            self.cap = cv2.VideoCapture(self.video_file)
        except:
            raise FileNotFoundError(f"Video file or capture device not found: {self.video_file}")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __del__(self):
        if self.cap is not None:
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
                self.face_detected = True
                self.face = face
                self.count_no_face = 0
            else:
                self.count_no_face += 1
                self.face_detected = False
        
        if self.count_no_face > 10:
            self.reset()
            return None
        
        return self.face


class SignalPlotter:
    def __init__(self, fs, plot_duration):
        self.fs = fs
        self.plot_duration = plot_duration
        self.setup_plot()

    def setup_plot(self):
        self.root = tk.Tk()
        self.root.title("Real-time Vital Signs")

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        self.ax1.set_title('BVP Signal | HR: -- bpm')
        self.ax2.set_title('RSP Signal | RR: -- br/min')
        self.line_bvp, = self.ax1.plot([], [])
        self.line_rsp, = self.ax2.plot([], [])

    def update_plot(self, bvp_data, rsp_data, hr, rr):
        if not hasattr(self, 'root'):
            return

        # Update BVP
        self.ax1.clear()
        self.ax1.plot(bvp_data)
        self.ax1.set_title(f'BVP Signal | HR: {hr:.0f} bpm')

        # Update RSP
        self.ax2.clear()
        self.ax2.plot(rsp_data)
        self.ax2.set_title(f'RSP Signal | RR: {rr:.0f} br/min')

        self.fig.tight_layout()
        self.canvas.draw()
        self.root.update()


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
        
        # Initialize plotter
        self.plotter = SignalPlotter(self.fs, self.plot_duration)
        
        # Setup output files
        config_stem = Path(config_path).stem
        # Extract video name, append timestamp for live video
        if type(self.video_file) == type(Path.home()):
            root_path = self.video_file.parent
            vid_name = self.video_file.stem
        else:
            # Default to Downloads folder
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
        
        while not self.stop_event.is_set() and count_frame < self.max_frames:
            frame = self.video_processor.get_frame()
            if frame is None:
                break
                
            if count_frame % self.face_detection_interval == 0:
                face = self.face_detector.detect_face(frame)

            if face is not None:
                x1, y1, x2, y2 = face
                frame = frame[y1:y2, x1:x2]
                frame = cv2.resize(frame, (self.width, self.height))
                frame = frame[np.newaxis, :, :, :]
                frame = frame.transpose(0, 3, 1, 2)
                self.frame_queue.put((count_frame, frame))
            else:
                print('Face not detected')
            
            count_frame += 1
            
        self.frame_queue.put(None)

    def inference_thread(self):
        count_frame = 0
        
        while not self.stop_event.is_set():
            data = self.frame_queue.get()
            if data is None:
                break
                
            frame_idx, frame = data
            
            if count_frame < self.num_frames:
                self.inference_frame_buffer[:, :, count_frame, :, :] = frame
            else:
                self.inference_frame_buffer[:, :, :-1, :, :] = self.inference_frame_buffer[:, :, 1:, :, :]
                self.inference_frame_buffer[:, :, -1, :, :] = frame
                
            if count_frame >= self.num_frames and count_frame % self.inference_interval == 0:
                t1 = time.time()
                bvp, rsp = self.model_instance.infer_rphys(self.inference_frame_buffer)
                t2 = time.time()
                print(f'Inference time: {t2-t1}')

                bvp, rsp = self.signal_processor.post_process(bvp, rsp)
                bvp = bvp.reshape(-1)
                rsp = rsp.reshape(-1)
                
                self.result_queue.put((bvp, rsp))
                
            count_frame += 1
            
        self.result_queue.put(None)

    def plotting_thread(self):
        while not self.stop_event.is_set():
            data = self.result_queue.get()
            if data is None:
                break
                
            bvp, rsp = data
            
            if len(self.estimated_bvp) == 0:
                self.estimated_bvp = bvp
                self.estimated_rsp = rsp
            else:
                self.estimated_bvp = np.concatenate([self.estimated_bvp, bvp[-self.inference_interval:]])
                self.estimated_rsp = np.concatenate([self.estimated_rsp, rsp[-self.inference_interval:]])
            
            plot_samples = int(self.plot_duration * self.fs)
            bvp_plot = self.estimated_bvp[-plot_samples:]
            rsp_plot = self.estimated_rsp[-plot_samples:]
            
            hr, rr = self.signal_processor.compute_metrics(bvp_plot, rsp_plot)
            self.hr_list.append(hr)
            self.rr_list.append(rr)
            
            self.plotter.update_plot(bvp_plot, rsp_plot, hr, rr)

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
            self.stop_event.set()

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
        finally:
            self.save_estimated_signals()
            self.save_plots()


def main():
    parser = argparse.ArgumentParser(description='Remote Vital Signs Detection')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    rPhysObj = RemoteVitalSigns(config, args.config)
    rPhysObj.workflow()

if __name__ == '__main__':
    main()

# tools/torch2onnx/SCAMPS_Multi_9x9.pth
# final_model_release/SCAMPS/SCAMPS_MMRPhysLEF_BVP_RSP_RGBx180x72_SFSAM_Label_Epoch0.pth
# final_model_release/BP4D_MMRPhysSEF_BVP_RSP_RGBx180x9_SFSAM_Label_Fold3_Epoch4.pth
