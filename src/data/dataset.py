import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import json

class WLASLDataset(Dataset):
    def __init__(self, json_path, video_dir, num_frames=16, img_size=224):
        """
        Args:
            json_path: Path to WLASL JSON file
            video_dir: Directory containing videos
            num_frames: Number of frames to sample from each video
            img_size: Size to resize frames to
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.img_size = img_size
        
        # Build list of (video_path, label) pairs
        self.samples = []
        for label_idx, sign in enumerate(self.data):
            gloss = sign['gloss']
            for instance in sign['instances']:
                video_id = instance['video_id']
                video_path = os.path.join(video_dir, f"{gloss}_{video_id}.mp4")
                if os.path.exists(video_path):
                    self.samples.append((video_path, label_idx, gloss))
        
        print(f"✅ Dataset initialized: {len(self.samples)} videos, {len(self.data)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label, gloss = self.samples[idx]
        
        # Load video
        frames = self.load_video(video_path)
        
        # Convert to tensor [T, C, H, W]
        frames = torch.from_numpy(frames).float()
        frames = frames / 255.0  # Normalize to [0, 1]
        
        return frames, label, gloss
    
    def load_video(self, video_path):
        """Load video and sample frames"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frame indices uniformly
        if total_frames < self.num_frames:
            # Repeat frames if video too short
            frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
        else:
            frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize and convert BGR to RGB
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        # Stack to [T, H, W, C]
        frames = np.stack(frames, axis=0)
        
        # Convert to [T, C, H, W]
        frames = frames.transpose(0, 3, 1, 2)
        
        return frames
