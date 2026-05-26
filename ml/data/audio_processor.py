import librosa
import numpy as np
import os
import subprocess
import cv2
import imageio_ffmpeg

class AudioProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    def extract_audio(self, video_path, audio_output_path):
        """
        Extracts audio from video using FFmpeg.
        """
        command = [
            self.ffmpeg_path, '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1',
            audio_output_path
        ]
        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except subprocess.CalledProcessError:
            print(f"Failed to extract audio from {video_path}")
            return False

    def audio_to_spectrogram(self, audio_path, image_output_path):
        """
        Converts a .wav file into a Mel-spectrogram and saves it as an image.
        """
        try:
            y, sr = librosa.load(audio_path, sr=None)
            
            # Generate Mel Spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # Normalize to 0-255 for image saving
            img = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8) * 255.0
            img = img.astype(np.uint8)
            
            # Apply a colormap for CNN visual feature learning
            img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            
            # Resize to target size (224x224)
            img_resized = cv2.resize(img_color, self.target_size)
            
            cv2.imwrite(image_output_path, img_resized)
            return True
        except Exception as e:
            print(f"Error generating spectrogram for {audio_path}: {e}")
            return False
