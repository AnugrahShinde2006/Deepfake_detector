import os
import sys
import torch
import cv2
import base64
from torchvision import transforms

# Ensure we can import from ml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.video_processor import VideoProcessor
from data.audio_processor import AudioProcessor
from models.visual_model import DeepfakeVisualModel
from models.audio_model import DeepfakeAudioModel
from models.temporal_model import DeepfakeTemporalModel
from models.fusion_model import DeepfakeFusionModel

class DeepfakePredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Data Processors
        self.vp = VideoProcessor(max_frames=10)
        self.ap = AudioProcessor()
        
        # Load Models (Visual model starts blank unless we find trained weights)
        self.visual_model = DeepfakeVisualModel(pretrained=False).to(self.device)
        
        # Check if the user has completed Phase 5 Training
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'training', 'weights', 'visual_best.pt')
        if os.path.exists(weights_path):
            print(f"Loading custom trained AI weights from {weights_path}...")
            self.visual_model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.visual_model.eval()
        else:
            print("WARNING: Custom AI weights not found. Using random weights. Please run train_visual.py!")
            self.visual_model.eval()

        self.audio_model = DeepfakeAudioModel(pretrained=False).to(self.device)
        self.temporal_model = DeepfakeTemporalModel().to(self.device)
        self.fusion_model = DeepfakeFusionModel().to(self.device)
        
        self.visual_model.eval()
        self.audio_model.eval()
        self.temporal_model.eval()
        self.fusion_model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def analyze_video(self, video_path):
        temp_faces_dir = os.path.join(os.path.dirname(video_path), "temp_faces")
        temp_audio_wav = os.path.join(os.path.dirname(video_path), "temp_audio.wav")
        temp_audio_spec = os.path.join(os.path.dirname(video_path), "temp_spec.jpg")
        
        # 1. Extraction
        faces = self.vp.extract_faces_from_video(video_path, temp_faces_dir)
        has_audio = self.ap.extract_audio(video_path, temp_audio_wav)
        if has_audio:
            has_audio = self.ap.audio_to_spectrogram(temp_audio_wav, temp_audio_spec)
            
        # 2. Visual Prediction
        visual_score = 0.5
        features_list = []
        base64_faces = []
        if len(faces) > 0:
            scores = []
            with torch.no_grad():
                for face_path in faces:
                    img = cv2.imread(face_path)
                    
                    # Convert to base64 for the frontend
                    _, buffer = cv2.imencode('.jpg', img)
                    b64_string = base64.b64encode(buffer).decode('utf-8')
                    base64_faces.append(f"data:image/jpeg;base64,{b64_string}")
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    # Extract 1280-dim feature vector for temporal model
                    features = self.visual_model.model.features(tensor)
                    features = self.visual_model.model.avgpool(features).flatten(1)
                    features_list.append(features)
                    
                    logit = self.visual_model(tensor)
                    prob = torch.sigmoid(logit).item()
                    scores.append(prob)
            visual_score = sum(scores) / len(scores)
            
        # 3. Audio Prediction
        audio_score = 0.5
        if has_audio and os.path.exists(temp_audio_spec):
            img = cv2.imread(temp_audio_spec)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logit = self.audio_model(tensor)
                audio_score = torch.sigmoid(logit).item()
                
        # 4. Temporal Prediction
        temporal_score = 0.5
        if len(features_list) > 0:
            seq_tensor = torch.stack(features_list, dim=1) # shape: (1, frames, 1280)
            with torch.no_grad():
                logit = self.temporal_model(seq_tensor)
                temporal_score = torch.sigmoid(logit).item()
                
        # 5. Fusion Prediction
        v_tensor = torch.tensor([[visual_score]], device=self.device)
        a_tensor = torch.tensor([[audio_score]], device=self.device)
        t_tensor = torch.tensor([[temporal_score]], device=self.device)
        
        with torch.no_grad():
            final_logit = self.fusion_model(v_tensor, a_tensor, t_tensor)
            final_score = torch.sigmoid(final_logit).item()
            
        return {
            "overall_score": final_score,
            "visual_score": visual_score,
            "audio_score": audio_score,
            "temporal_score": temporal_score,
            "is_fake": final_score > 0.5,
            "extracted_faces": base64_faces
        }
