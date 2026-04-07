# dataset.py
import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
import decord
from decord import VideoReader

decord.bridge.set_bridge('torch')

class SurgicalVideoTextDataset(Dataset):
    def __init__(self, root_dir, text_model_name, num_frames=8, image_size=224, csv_cache_path="lemon_dataset_cache.csv"):
        self.root_dir = root_dir
        self.csv_cache_path = csv_cache_path
        
        self.data = self._build_or_load_csv()
        
        self.num_frames = num_frames
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        self.transform = transforms.Compose([
            # 1. 随机裁剪并调整大小（模拟缩放不变性）
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), antialias=True),

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def _build_or_load_csv(self):
        if os.path.exists(self.csv_cache_path):
            print(f"[Dataset] Loading cached index from {self.csv_cache_path}")
            return pd.read_csv(self.csv_cache_path)

        print(f"[Dataset] Scanning directory {self.root_dir} to build index...")
        records = []
        
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
                
            for file in os.listdir(subdir_path):
                if file.endswith(".mp4"):
                    mp4_path = os.path.join(subdir_path, file)
                    txt_path = mp4_path.replace(".mp4", ".txt")
                    
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                text_content = f.read().strip()
                        except Exception as e:
                            continue
                            
                        records.append({
                            'video_path': mp4_path,
                            'text': text_content,
                            'confidence_weight': 1.0 
                        })

        df = pd.DataFrame(records)
        df.to_csv(self.csv_cache_path, index=False)
        print(f"[Dataset] Built and saved index to {self.csv_cache_path} (Total samples: {len(df)})")
        return df

    def __len__(self):
        return len(self.data)

    def _sample_frames(self, video_path):
        vr = VideoReader(video_path, num_threads=1)
        vlen = len(vr)
        if vlen == 0:
            raise ValueError(f"Video {video_path} has 0 frames.")
            
        indices = torch.linspace(0, vlen - 1, self.num_frames).long()
        frames = vr.get_batch(indices) 
        frames = frames.permute(0, 3, 1, 2).float() / 255.0 
        frames = self.transform(frames)
        return frames

    def __getitem__(self, idx):
        while True:
            try:
                row = self.data.iloc[idx]
                pixel_values = self._sample_frames(row['video_path'])
                
                text_inputs = self.tokenizer(
                    row['text'], 
                    padding='max_length', 
                    truncation=True, 
                    max_length=77, 
                    return_tensors="pt"
                )
                
                weight = torch.tensor(row['confidence_weight'], dtype=torch.float32)

                return {
                    'pixel_values': pixel_values,
                    'input_ids': text_inputs['input_ids'].squeeze(0),
                    'attention_mask': text_inputs['attention_mask'].squeeze(0),
                    'weight': weight
                }
                
            except Exception as e:
                idx = random.randint(0, len(self.data) - 1)