# SurgLIME



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
   $ git clone git@github.com:visurg-ai/PL-Stitch.git
   $ cd PL-Stitch && pip install -r requirements.txt
   ```
