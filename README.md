# Can LLM-Generated Text Empower Surgical Vision-Language Pre-training? (CVPRW 2026 Oral)


[![Paper](https://img.shields.io/badge/📚_Paper-arXiv-b31b1b?style=flat)](https://arxiv.org/abs/2604.18134)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-LIME-ffcc00?style=flat)](https://huggingface.co/datasets/visurg/LIME/tree/main)
[![Model](https://img.shields.io/badge/🤗_Model-SurgLIME-ffcc00?style=flat)](https://huggingface.co/visurg/SurgLIME/blob/main/surglime.pth)




This is the official repository for the CVPRW 2026 Oral paper "Can LLM-Generated Text Empower Surgical Vision-Language Pre-training?".


Star ⭐ us if you like it!


<img src="https://github.com/user-attachments/assets/4d397e7e-3262-43ba-b970-c6213d5171c4" />




Abstract
--------
Recent advancements in self-supervised learning have led to powerful surgical vision encoders capable of spatiotemporal understanding. However, extending these visual foundations to multi-modal reasoning tasks is severely bottlenecked by the prohibitive cost of expert textual annotations. To overcome this scalability limitation, we introduce \textbf{LIME}, a large-scale multi-modal dataset derived from open-access surgical videos using human-free, Large Language Model (LLM)-generated narratives. While LIME offers immense scalability, unverified generated texts may contain errors, including hallucinations, that could potentially lead to catastrophically degraded pre-trained medical priors in standard contrastive pipelines. To mitigate this, we propose \textbf{SurgLIME}, a parameter-efficient Vision-Language Pre-training (VLP) framework designed to learn reliable cross-modal alignments using noisy narratives. SurgLIME preserves foundational medical priors using a LoRA-adapted dual-encoder architecture and introduces an automated confidence estimation mechanism that dynamically down-weights uncertain text during contrastive alignment. Evaluations on the AutoLaparo and Cholec80 benchmarks show that SurgLIME achieves competitive zero-shot cross-modal alignment while preserving the robust linear probing performance of the visual foundation model.

<br>

🔧 Install dependencies
--------------------------------------------------

Install the following dependencies in your local setup:

   ```bash
   git clone git@github.com:visurg-ai/SurgLIME.git
   cd SurgLIME && pip install -r requirements.txt
   ```



🗂️ Preparation
-------------------
1. Download the pretraining dataset ([LIME](https://huggingface.co/datasets/visurg/LIME)) and evaluation datasets ([Cholec80](https://camma.unistra.fr/datasets/), [AutoLaparo](https://autolaparo.github.io/)).

2. Download the [PL-Stitch](https://github.com/visurg-ai/PL-Stitch) vision foundation model.

<img width="1102" height="548" alt="Screenshot 2026-04-08 at 10 09 50" src="https://github.com/user-attachments/assets/bf1c0987-e1a7-4eec-9e67-81296d40c51f" />


🚀 Training
-----------
We provide a script with default parameters for SurgLIME model pretraining.

1. Score the LLM-generated narratives from the LIME dataset using PubMedBERT:

```bash
python score_texts.py
```
2. Execute the SurgLIME pretraining script:
```bash
bash train.sh
```
3. Run zero-shot surgical phase recognition:
```bash
python zero_shot_autolaparo_LMDB.py
```


