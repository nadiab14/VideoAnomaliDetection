import torch
import torch.nn.functional as F
import cv2
import os
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import shutil
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import json 

# ==== CONFIG ==== 
RESULTS_DIR = r"E:\score__fclayers\results"
FRAME_SAVE_DIR = os.path.join(RESULTS_DIR, "clip_images")
FRAME_SIZE = (224, 224)
INITIAL_FRAME_STRIDE = 5
CLIP_LENGTH = 16
TOP_K = 4
clip_scores = []
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==== SETUP ==== 
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FRAME_SAVE_DIR, exist_ok=True)

# ==== Load CLIP ==== 
print("⏳ Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP loaded")

# === 2. Define your AnomalyDetector model architecture ===
class AnomalyDetector(torch.nn.Module):
    """
    A simple feedforward neural network to predict anomaly scores from CLIP embeddings.

    Architecture:
        - Linear(512 → 256) + ReLU + Dropout(0.3)
        - Linear(256 → 64) + ReLU
        - Linear(64 → 1) + Sigmoid

    Output:
        A value between 0 (normal) and 1 (abnormal).
    """
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    

# ==== Load Your Anomaly Model ==== 
model = AnomalyDetector().to(device)
model.load_state_dict(torch.load("detector3.pth", map_location=device))
model.eval()
print("✅ Anomaly detector model loaded")

# ==== Utility Functions ==== 
def get_video_info(video_path):
    """
    Extracts basic metadata from a video file.

    Parameters:
        video_path (str): Path to the input video.

    Returns:
        dict: Contains fps, total_frames, width, height, and duration.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    cap.release()
    return info

def get_clip_embeddings_batch(frame_pils):
    """
    Computes CLIP embeddings for a batch of PIL images.

    Parameters:
        frame_pils (List[PIL.Image]): List of resized PIL frames.

    Returns:
        torch.Tensor: CLIP image features (batch_size, 512).
    """
    inputs = clip_processor(images=frame_pils, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        return clip_model.get_image_features(**inputs)

def get_motion_score(prev_frame, curr_frame):
    """
    Computes motion score based on pixel difference between two frames.

    Parameters:
        prev_frame (np.ndarray): Previous frame.
        curr_frame (np.ndarray): Current frame.

    Returns:
        float: Motion score (higher means more motion).
    """
    diff = cv2.absdiff(prev_frame, curr_frame)
    score = np.sum(diff) / (diff.shape[0] * diff.shape[1])
    return score

def select_representative_frame(frame_buffer, avg_embedding):
    """
    Selects the frame in the buffer most different from the average embedding.

    Parameters:
        frame_buffer (List[PIL.Image]): List of frames in the clip.
        avg_embedding (torch.Tensor): Average CLIP embedding of the clip.

    Returns:
        PIL.Image: Most representative frame of the clip.
    """
    max_diff = -1
    rep_frame = None
    for frame in frame_buffer:
        frame_embedding = get_clip_embeddings_batch([frame]).mean(dim=0)
        diff = 1 - F.cosine_similarity(frame_embedding, avg_embedding, dim=0).item()
        if diff > max_diff:
            max_diff = diff
            rep_frame = frame
    return rep_frame

def extract_and_select_top_clips(video_path):
    """
    Extracts clips from a video, selects representative frames for each,
    and returns the top-K diverse clips based on CLIP embedding distances.

    Parameters:
        video_path (str): Path to the input video.

    Returns:
        List[dict]: Each dict contains clip_num, path, and embedding of top-K clips.
    """
    shutil.rmtree(FRAME_SAVE_DIR, ignore_errors=True)
    os.makedirs(FRAME_SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_info = get_video_info(video_path)
    total_frames = video_info["total_frames"]

    all_metadata = []
    frame_buffer = []
    frame_index = 0
    clip_index = 0
    prev_frame = None
    frame_stride = INITIAL_FRAME_STRIDE

    with tqdm(total=total_frames, desc="🎞️ Extracting clips ") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_stride == 0:
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(FRAME_SIZE)
                frame_buffer.append(frame_pil)

                if prev_frame is not None:
                    motion_score = get_motion_score(prev_frame, frame)
                    frame_stride = 3 if motion_score > 1000 else INITIAL_FRAME_STRIDE
                prev_frame = frame

                if len(frame_buffer) == CLIP_LENGTH:
                    embeddings = get_clip_embeddings_batch(frame_buffer)
                    avg_embedding = embeddings.mean(dim=0)
                    rep_frame = select_representative_frame(frame_buffer, avg_embedding)

                    clip_path = os.path.join(FRAME_SAVE_DIR, f"clip_{clip_index:03d}.jpg")
                    rep_frame.save(clip_path)

                    all_metadata.append({
                        "clip_num": clip_index,
                        "path": clip_path,
                        "embedding": avg_embedding.cpu()
                    })
                    clip_index += 1
                    frame_buffer = []
            frame_index += 1
            pbar.update(1)
    cap.release()

    if not all_metadata:
        return []

    # Select Top-K Diverse Clips
    selected = [all_metadata[0]]
    used = {0}
    for _ in range(1, min(TOP_K, len(all_metadata))):
        max_dist = -1
        best_idx = -1
        for i, meta in enumerate(all_metadata):
            if i in used:
                continue
            dists = [1 - F.cosine_similarity(meta["embedding"], s["embedding"], dim=0).item() for s in selected]
            min_dist = min(dists)
            if min_dist > max_dist:
                max_dist = min_dist
                best_idx = i
        if best_idx != -1:
            selected.append(all_metadata[best_idx])
            used.add(best_idx)

    return selected

def extract_features(image_path):
    """
    Extracts a single CLIP embedding from an image path.

    Parameters:
        image_path (str): Path to the saved clip image.

    Returns:
        np.ndarray: Feature vector of shape (512,).
    """
    image = Image.open(image_path).convert("RGB").resize(FRAME_SIZE)
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        return clip_model.get_image_features(**inputs).squeeze().cpu().numpy()

def get_anomaly_scores(video_paths):
    results = {}
    for path in video_paths:
        video_name = os.path.basename(path)  # Get just the filename
        selected_clips = extract_and_select_top_clips(path)
        video_results = []
        
        for clip in selected_clips:
            clip_path = clip["path"]
            features = extract_features(clip_path)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                score = model(features_tensor).item()
            
            video_results.append({
                "video_name": video_name,  # Add video name here
                "clip_path": clip_path,
                "score": score
            })
        
        results[video_name] = video_results  # Use video_name as key
    return results

# ==== MAIN ==== 
if __name__ == "__main__":
    video_paths = [
        #r"C:\Users\hendi\Downloads\dog_barking.mp4",
        #r"C:\Users\hendi\Downloads\IronMan.mp4",
        #r"C:\Users\hendi\Downloads\Abuse002_x264.mp4",
        #r"E:\score__fclayers\boat.mp4",
        #r"E:\score__fclayers\RoadAccidents011_x264 (1).mp4"
        #r"E:\score__fclayers\RoadAccidents005_x264.mp4"
        #r"E:\score__fclayers\Stealing008_x264.mp4"
        r"E:\score__fclayers\Shooting004_x264.mp4"
        #r"E:\score__fclayers\Robbery029_x264.mp4"
        #r"E:\score__fclayers\Normal_Videos_417_x264.mp4"
        #r"E:\score__fclayers\Course Poursuite et Accident de Voiture Spectaculaire - Car Crash.mp4"
        #r"E:\score__fclayers\Caméra de vidéosurveillance - Sécurité Mania - Phénomène paranormal - round 2.mp4"
        #r"E:\score__fclayers\birthday.mp4"
        #r"E:\score__fclayers\Abuse001_x264 (1).mp4"
        #r"E:\score__fclayers\applausing.mp4"
        #r"E:\score__fclayers\Arrest006_x264.mp4"
        #r"E:\score__fclayers\Assault011_x264.mp4"
        #r"E:\score__fclayers\Burglary020_x264.mp4"
        #r"E:\score__fclayers\Explosion006_x264.mp4"
        #r"E:\score__fclayers\Fighting016_x264A.mp4"
        #r"E:\score__fclayers\Normal_Videos_745_x264.mp4"
        #r"E:\score__fclayers\Normal_Videos_015_x264 (1).mp4"
        #r"C:\Users\hendi\Downloads\skateboarding_dog.mp4"
        #r"C:\Users\hendi\Downloads\Arson002_x264.mp4"
        #r"C:\Users\hendi\Downloads\Arrest001_x264.mp4"
        #r"C:\Users\hendi\Downloads\Arrest007_x264.mp4"
        #r"C:\Users\hendi\Downloads\Arrest030_x264.mp4"
        #r"C:\Users\hendi\Downloads\Burglary033_x264.mp4"
        #r"C:\Users\hendi\Downloads\Ezviz H8C 4MP Alarm Sirene Warning.mp4"
        #r"C:\Users\hendi\Downloads\CCTV footage.Mall CCTV camera video cctv camera video no copyright stock video full HD 4K.mp4"
        #r"C:\Users\hendi\Downloads\Breath Of Nature - Free Adventure Background Music (Free Hiking Music For Mountain Videos).mp4"
        #r"C:\Users\hendi\Downloads\1-Minute Nature Background Sound.mp4"
        #r"C:\Users\hendi\Downloads\Abuse029_x264.mp4"
        #r"C:\Users\hendi\Downloads\Abuse017_x264.mp4"
        #r"C:\Users\hendi\Downloads\Fighting042_x264A.mp4"
        #r"C:\Users\hendi\Downloads\Explosion011_x264A.mp4"
    ]
    

    
    anomaly_results = get_anomaly_scores(video_paths)
    with open("anomaly_scores.json", "w") as f:
        json.dump(anomaly_results, f, indent=4)
    print("Anomaly scores saved to anomaly_scores.json with video names")