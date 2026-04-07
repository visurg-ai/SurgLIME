import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

@torch.no_grad()
def main():
    # 1. 配置路径与设备
    # 指向你刚才使用 ffmpeg 压缩出的目标文件夹，确保与 train.py 读取的目录完全一致
    root_dir = "/root/LIME" 
    output_csv = "lemonTXT_dataset_cache.csv"
    model_id = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 加载 PubMedBERT (带有 MLM 头)
    print(f"Loading tokenizer and MLM model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
    model.eval()

    # 3. 遍历目录，收集数据
    print(f"Scanning directory: {root_dir}")
    records = []
    
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
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
                        
                        records.append({
                            'video_path': mp4_path,
                            'text': text_content
                        })
                    except Exception as e:
                        print(f"Error reading {txt_path}: {e}")

    df = pd.DataFrame(records)
    total_samples = len(df)
    print(f"Found {total_samples} valid video-text pairs.")

    # 4. 批量计算重构损失 (Pseudo-Perplexity)
    batch_size = 128
    losses = []
    
    print("Calculating Pseudo-Perplexity scores...")
    for i in tqdm(range(0, total_samples, batch_size)):
        batch_texts = df['text'].iloc[i:i+batch_size].tolist()
        
        # 编码文本
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        ).to(device)
        
        # 将 input_ids 赋值给 labels，HuggingFace 会自动计算交叉熵重构损失
        inputs["labels"] = inputs["input_ids"]
        
        # 前向传播
        outputs = model(**inputs)
        
        # 提取 batch 内每个样本的独立 loss
        # outputs.loss 是 batch 的均值，我们要获取 token 级的 loss 进行按样本平均
        logits = outputs.logits # [B, Seq_Len, Vocab_Size]
        labels = inputs["labels"] # [B, Seq_Len]
        
        # 展平计算 CrossEntropy，reduction='none' 
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(logits.view(-1, model.config.vocab_size), labels.view(-1))
        token_losses = token_losses.view(labels.size()) # [B, Seq_Len]
        
        # 忽略 padding token (ID=0) 的 loss
        mask = (labels != tokenizer.pad_token_id).float()
        sample_losses = (token_losses * mask).sum(dim=1) / mask.sum(dim=1)
        
        losses.extend(sample_losses.cpu().tolist())

    df['raw_loss'] = losses

    # 5. 归一化为 Confidence Weight (0.1 ~ 1.0)
    # 损失越高 -> 幻觉/噪声越大 -> 权重越低
    min_loss = df['raw_loss'].min()
    max_loss = df['raw_loss'].max()
    
    # 线性反向映射: W_norm \in [0, 1]
    df['w_norm'] = 1.0 - (df['raw_loss'] - min_loss) / (max_loss - min_loss)
    
    # 缩放至 [0.1, 1.0]，防止在 InfoNCE 中除以 0 或彻底杀死梯度
    df['confidence_weight'] = df['w_norm'] * 0.9 + 0.1
    
    # 清理中间列
    df = df.drop(columns=['raw_loss', 'w_norm'])

    # 6. 保存输出
    df.to_csv(output_csv, index=False)
    print(f"Scoring complete. Dataset cache with weights saved to {output_csv}")
    print("Sample distribution of confidence weights:")
    print(df['confidence_weight'].describe())

if __name__ == "__main__":
    main()