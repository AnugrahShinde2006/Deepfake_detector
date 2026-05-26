import os
import zipfile
from huggingface_hub import hf_hub_download

def setup_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    faces_dir = os.path.join(base_dir, 'processed_data', 'faces')
    
    real_dir = os.path.join(faces_dir, 'real')
    fake_dir = os.path.join(faces_dir, 'fake')
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    print("Fetching Real Faces (wiki.zip)...")
    real_zip = hf_hub_download(repo_id="OpenRL/DeepFakeFace", filename="wiki.zip", repo_type="dataset")
    print(f"Unpacking to {real_dir}...")
    with zipfile.ZipFile(real_zip, 'r') as zip_ref:
        # Extract a subset to keep it fast
        file_list = zip_ref.namelist()[:2000] 
        for file in file_list:
            if file.endswith('.jpg') or file.endswith('.png'):
                zip_ref.extract(file, real_dir)
                
    print("Fetching Fake Faces (inpainting.zip)...")
    fake_zip = hf_hub_download(repo_id="OpenRL/DeepFakeFace", filename="inpainting.zip", repo_type="dataset")
    print(f"Unpacking to {fake_dir}...")
    with zipfile.ZipFile(fake_zip, 'r') as zip_ref:
        file_list = zip_ref.namelist()[:2000]
        for file in file_list:
            if file.endswith('.jpg') or file.endswith('.png'):
                zip_ref.extract(file, fake_dir)
                
    print("Dataset setup complete! Images are natively structured in processed_data/faces")

if __name__ == "__main__":
    setup_dataset()
