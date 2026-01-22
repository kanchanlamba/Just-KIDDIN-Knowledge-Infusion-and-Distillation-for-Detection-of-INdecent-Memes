# KID-VLM: Knowledge-Infused Distilled Vision-Language Model for Toxic Meme Detection

## 📌 Project Overview
This project addresses the challenge of detecting **toxic vs non-toxic memes** by jointly analyzing **visual and textual modalities**. Memes often convey harmful intent only through the interaction between image and text, making unimodal approaches insufficient.

We implement and benchmark multiple **text-only, image-only, and multimodal models**, culminating in a **Knowledge-Infused Distilled Vision-Language Model (KID-VLM)** inspired by recent neurosymbolic research. The project emphasizes **balanced evaluation, explainability, and efficient multimodal fusion** under limited computational resources.

---

## 🧠 Key Contributions
- Performed **extensive EDA and balanced downsampling** on the Hateful Memes dataset.
- Implemented **multiple baseline models** across text, image, and multimodal categories.
- Designed a **lightweight KID-VLM framework** integrating:
  - Knowledge Distillation (teacher–student paradigm)
  - External knowledge signals
  - Multimodal feature fusion
- Conducted **comprehensive evaluation** using Accuracy, Precision, Recall, F1-Score, and AUC.
- Analyzed **class-wise behavior**, highlighting recall–precision trade-offs in toxic meme detection.

---

## 📂 Dataset
**Hateful Memes Dataset (Facebook AI Research)**  
- Binary classification: *Toxic / Non-Toxic*
- Multimodal (Image + Text)
- Original size: ~8,500 memes  
- After cleaning & downsampling: **4,000 balanced samples**
- Images resized to 224×224 and text cleaned for consistency.

---

## 🧪 Models Implemented

### 🔹 Text-Only Models
- TF-IDF + SVM
- BiLSTM
- BERT (Fine-tuned)

### 🔹 Image-Only Models
- ResNet50 + Random Forest
- EfficientNetB0 + Random Forest
- Vision Transformer (ViT) + Random Forest

### 🔹 Multimodal Models
- CLIP + MLP
- Early Fusion (BERT + ResNet50)
- Late Fusion (BERT + ResNet50)
- **KID-VLM (Knowledge-Infused Distilled VLM)**

---

## 🧬 KID-VLM Architecture
- **Teacher Model:** CLIP (frozen)
- **Student Model:** Lightweight multimodal encoder
- **Knowledge Source:** Keyword-based contextual features
- **Fusion Strategy:** Feature concatenation + MLP
- **Loss:** Classification loss with teacher guidance

> KID-VLM aims to retain multimodal alignment while remaining computationally efficient.

---

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Learning Curves

---

## 📈 Key Observations
- Text-only BERT performs strongly due to contextual embeddings.
- Image-only models struggle with nuanced toxicity.
- Multimodal fusion significantly improves robustness.
- KID-VLM exhibits **extreme class-specific behavior** due to dataset reduction and fusion sensitivity, highlighting challenges in recall optimization.

---

## ⚠️ Limitations
- Reduced dataset size limits generalization.
- Class imbalance effects persist even after downsampling.
- Large multimodal transformers (VisualBERT, ViLBERT) could not be fully integrated due to resource constraints.

---

## 🚀 Future Work
- Improved class-balancing strategies (focal loss, adaptive sampling).
- Graph-based knowledge infusion (ConceptNet).
- Cross-modal attention visualization for explainability.
- Extension to multilingual and culturally diverse meme datasets.

---

## 🛠 Tech Stack
- **Languages:** Python
- **Frameworks:** PyTorch, HuggingFace Transformers
- **Vision Models:** CLIP, ResNet50, EfficientNetB0, ViT
- **NLP Models:** BERT, BiLSTM, TF-IDF
- **ML Libraries:** scikit-learn, NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Google Colab

---
