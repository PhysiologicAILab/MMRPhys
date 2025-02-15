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

from dataset.data_loader.face_detector.YOLO5Face import YOLO5Face
from tools.torch2onnx.MMRPhysSEF import MMRPhysSEF


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _calculate_fft_rate(phys_signal, fs=30, low_pass=0.6, high_pass=3.3):
    """Calculate heart rate/ resp rate using Fast Fourier transform (FFT)."""
    phys_signal = np.expand_dims(phys_signal, 0)
    N = phys_signal.shape[1]
    if N <= 30*fs:
        nfft = _next_power_of_2(N)
        f_phys, pxx_phys = periodogram(
            phys_signal, fs=fs, nfft=nfft, detrend=False)
    else:
        f_phys, pxx_phys = welch(phys_signal, fs=fs, nperseg=N//2, detrend=False)
    fmask_phys = np.argwhere((f_phys >= low_pass) & (f_phys <= high_pass))
    mask_phys = np.take(f_phys, fmask_phys)
    mask_pxx = np.take(pxx_phys, fmask_phys)
    fft_hr = np.take(mask_phys, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr



class MMRPhysSEFInference:
    def __init__(self, model_path, video_file):

        self.num_frames = 181
        self.in_channels = 3
        self.height = 9
        self.width = 9

        md_config = {}
        md_config["FRAME_NUM"] = 181
        md_config["TASKS"] = ["BVP", "RSP"]
        self.model = MMRPhysSEF(self.num_frames, md_config, self.in_channels)

        lowcut_bvp = 0.6
        highcut_bvp = 3.3
        lowcut_rsp = 0.1
        highcut_rsp = 0.5
        self.fs = 30
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

        # initialize YOLOv5 face detector
        self.face_detector = YOLO5Face()
        self.frame_interal_for_face_detection = 180

        # Load model
        self.model = ort.InferenceSession(model_path)

        self.frame_buffer = np.zeros(
            (1, self.in_channels, self.num_frames, self.height, self.width))
        self.estimated_bvp = np.array([])
        self.estimated_rsp = np.array([])

        self.video_file = Path(video_file)
        self.json_filename = self.video_file.parent.joinpath(self.video_file.stem + '_estimated_signals_onnx.json')
        self.plot_filename = self.video_file.parent.joinpath(self.video_file.stem + '_estimated_signals_onnx.png')

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

    def infer_rphys(self):

        output = self.model.run(None, {'input': self.frame_buffer.astype(np.float32)})

        bvp = output[0]
        rsp = output[1]

        # apply normalization and bandpass filter
        bvp = (bvp - np.mean(bvp))/np.std(bvp)
        rsp = (rsp - np.mean(rsp))/np.std(rsp)
        bvp = signal.filtfilt(self.b_bvp, self.a_bvp, bvp)
        rsp = signal.filtfilt(self.b_rsp, self.a_rsp, rsp)

        return bvp, rsp


    def infer_from_file(self):
        '''
        load video file and return face-cropped frames
        run face detection and inference after every self.frame_interal_for_face_detection frames
        '''
        cap = cv2.VideoCapture(self.video_file)
        
        face_detected = False
        run_inference = False
        count_frame = 0
 
        x1, y1, x2, y2 = None, None, None, None     

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # detect face after every self.frame_interal_for_face_detection frames
            if count_frame % self.frame_interal_for_face_detection == 0 or not face_detected:
                face = self.face_detector.detect_face(frame)
                if face is None:
                    face_detected = False
                else:
                    face_detected = True
                    x1, y1, x2, y2 = face
            if x1 is None:
                continue
            frame = frame[y1:y2, x1:x2]
            frame = cv2.resize(frame, (self.width, self.height))
            frame = frame[np.newaxis, :, :, :]
            frame = frame.transpose(0, 3, 1, 2)

            if count_frame < self.num_frames:
                self.frame_buffer[:, :, count_frame , :, :] = frame
            else:
                self.frame_buffer[:, :, :-1, :, :] = self.frame_buffer[:, :, 1:, :, :]
                self.frame_buffer[:, :, -1, :, :] = frame

            if not run_inference and count_frame >= self.num_frames:
                run_inference = True
                
            if run_inference and count_frame % self.frame_interal_for_face_detection == 0:

                bvp, rsp = self.infer_rphys()

                bvp = bvp.reshape(-1)
                rsp = rsp.reshape(-1)

                # append estimated signals
                self.estimated_bvp = np.concatenate(
                    (self.estimated_bvp[self.frame_interal_for_face_detection:], bvp), axis=0)
                self.estimated_rsp = np.concatenate(
                    (self.estimated_rsp[self.frame_interal_for_face_detection:], rsp), axis=0)

            count_frame += 1

        cap.release()

    def compute_metrics(self):
        # compute heart rate
        self.hr = _calculate_fft_rate(self.estimated_bvp, self.fs, low_pass=0.6, high_pass=3.3)
        # validate heart rate
        if self.hr < 40 or self.hr > 200:
            self.hr = np.nan
        else:
            self.hr = np.round(self.hr, 0)
        print(f'Heart rate: {self.hr}')

        # Compute respiration rate
        self.rr = _calculate_fft_rate(self.estimated_rsp, self.fs, low_pass=0.1, high_pass=0.5)
        # validate respiration rate
        if self.rr < 5 or self.rr > 40:
            self.rr = np.nan
        else:
            self.rr = np.round(self.rr, 0)
        print(f'Resp rate: {self.rr}')


    def save_estimated_signals(self):
        # save estimated signals in a dictionary
        estimated_signals = {}
        estimated_signals['hr'] = self.hr
        estimated_signals['rr'] = self.rr
        estimated_signals['bvp'] = self.estimated_bvp.tolist()
        estimated_signals['rsp'] = self.estimated_bvp.tolist()

        # save estimated signals in json file
        with open(self.json_filename, 'w') as f:
            json.dump(estimated_signals, f)


    def save_plots(self):
        # Make combines plots of estimated signals
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.estimated_bvp)
        plt.title('Estimated BVP | HR: {:.2f}'.format(self.hr))
        plt.subplot(2, 1, 2)
        plt.plot(self.estimated_rsp)
        plt.title('Estimated RSP | RR: {:.2f}'.format(self.rr))
        plt.tight_layout()
        plt.savefig(self.plot_filename)
        plt.close()

    def workflow(self):
        self.infer_from_file()
        self.compute_metrics()
        self.save_estimated_signals()
        self.save_plots()


def main():
    parser = argparse.ArgumentParser(description='Inference on video file')
    parser.add_argument('--model_path', type=str, 
                        default='tools/torch2onnx/SCAMPS_Multi_9x9.onnx', help='path to model file')
    parser.add_argument('--video_file', type=str, 
                        default='/Users/jiteshjoshi/Downloads/MMRPhys_Results/webapp/recoded_video.mov', help='path to video file')
    args = parser.parse_args()

    model_path = args.model_path
    video_file = args.video_file

    rPhysObj = MMRPhysSEFInference(model_path, video_file)
    rPhysObj.workflow()

if __name__ == '__main__':
    main()