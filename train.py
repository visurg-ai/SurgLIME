# train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import get_cosine_schedule_with_warmup

from model import SurgLIME
from dataset import SurgicalVideoTextDataset

# --------------------------------------------------------
# 可导的跨 GPU 特征收集 (DDP InfoNCE 的核心)
# --------------------------------------------------------
class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def gather_features(tensor):
    if dist.is_available() and dist.is_initialized():
        tensors_gather = GatherLayer.apply(tensor)
        return torch.cat(tensors_gather, dim=0)
    return tensor

def confidence_weighted_infonce(v_proj, t_proj, logit_scale, weights):
    v_proj_all = gather_features(v_proj)
    t_proj_all = gather_features(t_proj)

    logits_per_video = logit_scale * v_proj @ t_proj_all.T
    logits_per_text = logit_scale * t_proj @ v_proj_all.T
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_batch_size = v_proj.size(0)
    labels = torch.arange(local_batch_size, dtype=torch.long, device=v_proj.device)
    labels = labels + rank * local_batch_size

    loss_v = F.cross_entropy(logits_per_video, labels, reduction='none')
    loss_t = F.cross_entropy(logits_per_text, labels, reduction='none') 

    loss_v_weighted = (loss_v * weights).sum() / local_batch_size
    loss_t_weighted = (loss_t * weights).sum() / local_batch_size
    
    return (loss_v_weighted + loss_t_weighted) / 2

# --------------------------------------------------------
# 核心训练逻辑
# --------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Parameter-Efficient Surgical VLP")
    parser.add_argument("--local_rank", type=int, default=-1)
    
    # 路径参数
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_surgvlp", help="Save directory")
    parser.add_argument("--vision_ckpt_path", type=str, required=True, help="Path to your pretrained ViT .pth file")
    
    # 模型架构参数
    parser.add_argument("--text_model_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=8)
    
    # 优化与保存参数
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--proj_lr_multiplier", type=float, default=10.0) 
    # [新增] 动态控制预热周期的参数，支持浮点数（如 1.5 个 epoch）
    parser.add_argument("--warmup_epochs", type=float, default=1.0, help="Number of epochs for learning rate warmup")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_interval", type=int, default=5, help="Save full checkpoint every N epochs")
    
    return parser.parse_args()

def main():
    args = parse_args()

    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    
    is_distributed = args.local_rank != -1
    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SurgLIME(
        text_model_name=args.text_model_name,
        vision_ckpt_path=args.vision_ckpt_path,
        proj_dim=args.proj_dim,
        lora_r=args.lora_r
    ).to(device)

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=False 
        )

    dataset = SurgicalVideoTextDataset(
        root_dir=args.root_dir,
        text_model_name=args.text_model_name,
        num_frames=args.num_frames,
        csv_cache_path="lemonTXT_dataset_cache.csv" 
    )
    
    sampler = DistributedSampler(dataset) if is_distributed else None
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=(sampler is None), 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )

    model_without_ddp = model.module if is_distributed else model
    
    lora_params = []
    proj_params = []
    
    for name, param in model_without_ddp.named_parameters():
        if param.requires_grad:
            if "vision_proj" in name or "text_proj" in name:
                proj_params.append(param)
            else:
                lora_params.append(param)
                
    optimizer_grouped_parameters = [
        {"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": proj_params, "lr": args.lr * args.proj_lr_multiplier, "weight_decay": args.weight_decay}
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    scaler = torch.amp.GradScaler('cuda') 

    # [核心修改] 根据 args.warmup_epochs 计算 warmup_steps
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(len(dataloader) * args.warmup_epochs) 
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    is_master = (not is_distributed) or (dist.get_rank() == 0)
    if is_master and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for epoch in range(args.epochs):
        if is_distributed:
            sampler.set_epoch(epoch)
            
        model.train()
        total_loss = 0.0
        
        if is_master:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = dataloader
            
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            weights = batch['weight'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                v_proj, t_proj, logit_scale = model(pixel_values, input_ids, attention_mask)
                loss = confidence_weighted_infonce(v_proj, t_proj, logit_scale, weights)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            with torch.no_grad():
                model_without_ddp.logit_scale.clamp_(0, 4.6052)
                
            total_loss += loss.item()
            if is_master:
                current_lr_base = scheduler.get_last_lr()[0]
                current_lr_proj = scheduler.get_last_lr()[1]
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'lr_b': f"{current_lr_base:.1e}", 
                    'lr_p': f"{current_lr_proj:.1e}"
                })
                
        if is_master:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}")
            
            full_state_dict = model_without_ddp.state_dict()
            
            latest_path = os.path.join(args.save_dir, "latest.pth")
            torch.save(full_state_dict, latest_path)
            
            if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
                save_path = os.path.join(args.save_dir, f"epoch_{epoch+1}.pth")
                torch.save(full_state_dict, save_path)
                print(f"Full checkpoint saved to {save_path}")

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()