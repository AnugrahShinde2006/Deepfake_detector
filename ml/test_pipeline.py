import os
from data.video_processor import VideoProcessor
from data.audio_processor import AudioProcessor

def test():
    # Setup test directories
    test_video = "sample.mp4"
    output_faces = "processed_data/faces"
    output_audio = "processed_data/audio"
    
    os.makedirs(output_faces, exist_ok=True)
    os.makedirs(output_audio, exist_ok=True)
    
    if not os.path.exists(test_video):
        print(f"Please place a '{test_video}' in the ml/ directory to test.")
        return
        
    print("Testing Video Processor (Face Extraction)...")
    vp = VideoProcessor(max_frames=5)
    faces = vp.extract_faces_from_video(test_video, output_faces)
    print(f"Extracted {len(faces)} faces.")
    
    print("Testing Audio Processor (Spectrogram Generation)...")
    ap = AudioProcessor()
    wav_path = os.path.join(output_audio, "temp.wav")
    spec_path = os.path.join(output_audio, "spectrogram.jpg")
    
    if ap.extract_audio(test_video, wav_path):
        if ap.audio_to_spectrogram(wav_path, spec_path):
            print("Successfully generated Mel-Spectrogram.")
        else:
            print("Spectrogram generation failed.")
    else:
        print("Audio extraction failed.")
        
if __name__ == "__main__":
    test()
