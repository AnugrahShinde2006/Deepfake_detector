import cv2
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN

class VideoProcessor:
    def __init__(self, target_size=(224, 224), max_frames=300):
        self.target_size = target_size
        self.max_frames = max_frames
        
        # Initialize MTCNN on the GPU for maximum performance
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # margin=20 adds padding around the detected face for better context
        self.mtcnn = MTCNN(keep_all=False, device=self.device, margin=20, post_process=False)

    def extract_faces_from_video(self, video_path, output_dir):
        """
        Extracts up to `max_frames` uniform frames from a video, detects the face using MTCNN, 
        crops it, and saves it to the output_dir.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []

        interval = max(1, total_frames // self.max_frames)
        
        extracted_paths = []
        frame_idx = 0
        
        # --- CPU PHASE ---
        # Uncompress the video using Integrated Graphics and load 10 frames into RAM
        frames_batch = []
        saved_count = 0

        while cap.isOpened() and saved_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                # MTCNN expects RGB images
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_batch.append(frame_rgb)
                saved_count += 1
            
            frame_idx += 1

        cap.release()
        
        if len(frames_batch) == 0:
            return []

        # --- GPU PHASE ---
        # Throw the entire batch of 10 frames at the RTX 3050 simultaneously!
        try:
            # boxes_batch is a list of bounding boxes for each of the 10 frames
            boxes_batch, probs_batch = self.mtcnn.detect(frames_batch)
        except Exception as e:
            print(f"MTCNN Batch Detection failed: {e}")
            return []

        # --- CPU POST-PROCESSING PHASE ---
        # Crop and save the detected faces to disk
        for i, (boxes, frame_rgb) in enumerate(zip(boxes_batch, frames_batch)):
            if boxes is not None and len(boxes) > 0:
                # Take the highest probability face (first one)
                x1, y1, x2, y2 = [int(b) for b in boxes[0]]
                
                # Ensure coordinates are within image bounds
                ih, iw, _ = frame_rgb.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw, x2), min(ih, y2)
                
                face_crop = frame_rgb[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    face_resized = cv2.resize(face_crop, self.target_size)
                    # Convert back to BGR so OpenCV saves the colors correctly
                    face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
                    
                    base_name = os.path.basename(video_path).split('.')[0]
                    save_path = os.path.join(output_dir, f"{base_name}_frame_{i}.jpg")
                    cv2.imwrite(save_path, face_bgr)
                    extracted_paths.append(save_path)

        return extracted_paths
