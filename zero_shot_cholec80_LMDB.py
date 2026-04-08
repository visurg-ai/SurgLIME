import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer

# Import model and dataset dependencies
from model import SurgicalVLP
from load_lmdb import Dataset as LMDBDataset, StringToIndexTransform, SubsetDataset

# -------------------------
# Config & Prompts
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path Configurations
CHECKPOINT_PATH = "SurgLIME.pth"
VISION_CKPT_PATH = "PL-Stitch.pth" 
TEXT_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

data_lmdb = "cholecT80_phase_recognition.lmdb/"  
label_lmdb = "cholecT80_phase_recognition.json"

BATCH_SIZE = 100 
NUM_WORKERS = 8
IMG_SIZE = 224
NUM_FRAMES = 8

class_mapping = {
    'Preparation': 0, 'CalotTriangleDissection': 1, 'ClippingCutting': 2, 
    'GallbladderDissection': 3, 'GallbladderPackaging': 4, 
    'CleaningCoagulation': 5, 'GallbladderRetraction': 6
}

CHOLEC80_PROMPTS = [
    "Prepares for surgery by inserting trocars into the patient's abdominal cavity",
    "Employs grasper and hook during calot triangle dissection, manipulating galdbladder to reveal hepatic triangle, cystic duct and cystic artery",
    "Utilizes clipper to secure cystic duct and artery, followed by precise dissection using scissors",
    "Utilizes a hook to dissect the connective tissue during the dissection phase, separating gallbladder from the liver",
    "Secures the removed gallbladder in the specimen bag during the packaging phase of the procedure",
    "Employs suction and irrigation techniques to maintain a clear surgical field during the clean and coagulation phase, simultaneously coagulating bleeding vessels",
    "Handles the specimen bag during the retraction phases, carefully extracting it from the trocar"
]

# -------------------------
# Temporal Dataset Wrapper (with cross-video truncation)
# -------------------------
class TemporalDatasetWrapper(Dataset):
    def __init__(self, base_subset, num_frames=8):
        self.base_subset = base_subset
        self.num_frames = num_frames
        self.length = len(base_subset)
        self.offsets = list(range(-(num_frames // 2 - 1), num_frames // 2 + 1))
        
        # Prefetch all keys for boundary checks
        self.keys = []
        original_dataset = self.base_subset.dataset
        subset_indices = self.base_subset.indices
        
        # original_dataset.index_img() should return a list of tuples: [('video41_00000.png', label), ...]
        all_imgs = original_dataset.index_img()
        for i in range(self.length):
            orig_idx = subset_indices[i]
            key_name = all_imgs[orig_idx][0]
            self.keys.append(key_name)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        _, center_label = self.base_subset[idx]
        
        # Parse the video name of the current frame, e.g., "video41"
        center_video_name = self.keys[idx].split('_')[0]

        frames = []
        for offset in self.offsets:
            target_idx = max(0, min(self.length - 1, idx + offset))
            
            # Strict boundary check: If the offset frame belongs to a different video, 
            # force the use of the center frame (or valid boundary) for padding.
            target_video_name = self.keys[target_idx].split('_')[0]
            if target_video_name != center_video_name:
                target_idx = idx 
                
            img, _ = self.base_subset[target_idx]
            frames.append(img)

        video_tensor = torch.stack(frames, dim=0)
        
        # Return center_video_name for subsequent Video-wise Metric calculation
        return video_tensor, center_label, center_video_name

# -------------------------
# Load Dataset 
# -------------------------
def load_test_dataloader(lmdb_path, label_path, class_map, transform):
    print('Loading dataset index...')
    train_ds = LMDBDataset(lmdb_path=lmdb_path, label_path=label_path, target_transform=StringToIndexTransform(class_map))
    
    num_train = len(train_ds)
    indx = train_ds.index_img()
    class_ids = {f'video{class_idx:02d}': [] for class_idx in range(1, 81)}

    for idx in range(num_train):
        idx_name = indx[idx][0]
        class_ids[idx_name.split('_')[0]].append(idx)

    video_ids = list(class_ids.keys())
    
    test_set_indices = []
    for class_idx in video_ids[40:]:
        test_set_indices += class_ids[class_idx]

    valid_dataset = SubsetDataset(dataset=train_ds, indices=test_set_indices, transform=transform)
    temporal_valid_dataset = TemporalDatasetWrapper(valid_dataset, num_frames=NUM_FRAMES)
    
    test_loader = DataLoader(
        temporal_valid_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    return test_loader

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

test_loader = load_test_dataloader(data_lmdb, label_lmdb, class_mapping, transform)

# -------------------------
# Load Model & Weights
# -------------------------
print("\nBuilding SurgicalVLP model...")
model = SurgicalVLP(
    text_model_name=TEXT_MODEL_NAME,
    vision_ckpt_path=VISION_CKPT_PATH, 
    proj_dim=256,
    lora_r=16
).to(device)

print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
msg = model.load_state_dict(state_dict, strict=True)
print('Loaded full weights!!!', msg)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

# -------------------------
# Step 1: Encode Text Prompts
# -------------------------
@torch.no_grad()
def get_text_embeddings(prompts, model, tokenizer):
    print("Encoding text prompts (Zero-Shot Classifiers)...")
    inputs = tokenizer(prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    text_outputs = model.text_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    t_raw = text_outputs.last_hidden_state[:, 0, :] 
    t_proj = F.normalize(model.text_proj(t_raw), p=2, dim=-1)
    return t_proj

text_embeddings = get_text_embeddings(CHOLEC80_PROMPTS, model, tokenizer)

# -------------------------
# Step 2: Extract Vision Features & Predict
# -------------------------
@torch.no_grad()
def evaluate_zero_shot(dataloader, model, text_embeddings):
    all_preds, all_labels, all_videos = [], [], []
    
    print("\nRunning Zero-Shot Evaluation over Temporal Windows...")
    for pixel_values, labels, video_names in tqdm(dataloader):
        pixel_values = pixel_values.to(device, non_blocking=True)
        B, T, C, H, W = pixel_values.shape
        
        pixel_values_flat = pixel_values.view(B * T, C, H, W)
        v_cls_flat = model.vision_encoder(pixel_values_flat) 
        
        v_seq = v_cls_flat.view(B, T, -1)
        v_raw = model.temporal_pool(v_seq) 
        v_proj = F.normalize(model.vision_proj(v_raw), p=2, dim=-1) 
        
        logits = v_proj @ text_embeddings.T 
        preds = logits.argmax(dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        
        # Robust label extraction to handle Tensor/List/Tuple variations
        if isinstance(labels, torch.Tensor):
            all_labels.extend(labels.cpu().numpy())
        else:
            extracted_labels = [int(l.item()) if isinstance(l, torch.Tensor) else int(l) for l in labels]
            all_labels.extend(extracted_labels)
            
        all_videos.extend(video_names)
        
    return np.array(all_labels), np.array(all_preds), np.array(all_videos)

true_labels, predicted_labels, video_names = evaluate_zero_shot(test_loader, model, text_embeddings)

# -------------------------
# Step 3: Metrics (Frame-wise & Video-wise)
# -------------------------
# 1. Calculate Frame-wise (Global) Metrics
global_acc = accuracy_score(true_labels, predicted_labels)
global_macro_f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

# 2. Calculate Video-wise Metrics
unique_videos = np.unique(video_names)
video_accs = []
video_macro_f1s = []

for vid in unique_videos:
    # Extract all frames for the current video
    mask = (video_names == vid)
    vid_true = true_labels[mask]
    vid_pred = predicted_labels[mask]
    
    # Calculate Accuracy for the current video
    v_acc = accuracy_score(vid_true, vid_pred)
    video_accs.append(v_acc)
    
    # Calculate Macro-F1 for the current video
    v_f1 = f1_score(vid_true, vid_pred, average='macro', zero_division=0)
    video_macro_f1s.append(v_f1)

final_video_acc = np.mean(video_accs)
final_video_f1 = np.mean(video_macro_f1s)

# 3. Structured Print Output
print("\n" + "="*65)
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Comprehensive Zero-Shot Results")
print("="*65)
print(f"Total Test Videos Assessed : {len(unique_videos)}")
print(f"Total Test Frames Assessed : {len(true_labels)}")
print("-" * 65)
print(" [ Frame-wise (Global) Metrics ]")
print(f" Accuracy                  : {global_acc * 100:.2f}%")
print(f" Macro-avg F1              : {global_macro_f1 * 100:.2f}%")
print("-" * 65)
print(" [ Video-wise Metrics ]")
print(f" Accuracy                  : {final_video_acc * 100:.2f}%")
print(f" Macro-avg F1              : {final_video_f1 * 100:.2f}%")
print("="*65)

# 4. Print Detailed Class-wise Report (Frame-wise)
target_names = list(class_mapping.keys())
print("\nDetailed Frame-wise Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=target_names, zero_division=0))
print("="*65)