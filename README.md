# 🧠 Multilabel Text Classification

This repository explores various approaches to **Multilabel Text Classification**, where each input can belong to **multiple categories** simultaneously. We demonstrate this across three paradigms:

- 🔹 Traditional Machine Learning
- 🔹 Deep Learning with CNN
- 🔹 Transformer-based Fine-Tuning (BERT)

---

## 📚 Overview

Multilabel classification differs from multiclass classification in that **labels are not mutually exclusive**. For example, a toxic comment might be classified as both *toxic* and *insulting*.

This repo walks through:
- Preprocessing text for multilabel outputs
- Model architectures suitable for multilabel learning
- Metrics tailored for evaluating multilabel predictions

---

## 🧪 Datasets Used

| Approach | Dataset | Classes | Avg Labels/Sample | Format |
|---------|---------|---------|-------------------|--------|
| Logistic Regression | RCV1            | 103     | ~3                | `sklearn.datasets.fetch_rcv1` |
| CNN              | GoEmotions (simplified) | 28      | ~2                | `datasets.load_dataset("go_emotions", "simplified")` |
| BERT             | Jigsaw Toxic Comments   | 6       | ~1.8              | CSV (Kaggle) |

---

## 📁 Folder Structure

```

text-multilabel-classification/
├── notebooks/
│   ├── 1\_logreg\_rcv1\_multilabel.ipynb
│   ├── 2\_cnn\_goemotions\_multilabel.ipynb
│   └── 3\_bert\_jigsaw\_multilabel.ipynb
├── requirements.txt
└── README.md

````

---

## 🧰 Models

### 1️⃣ Logistic Regression + TF-IDF

- Wrapped in `OneVsRestClassifier`
- Vectorization: TF-IDF
- Loss Function: Logistic (sigmoid for multilabel)
- Evaluation: F1-score

### 2️⃣ CNN on GoEmotions

- Embedding layer + Conv1D + GlobalMaxPool
- Activation: `sigmoid`
- Loss: Binary Crossentropy
- Framework: TensorFlow / Keras

### 3️⃣ BERT on Jigsaw

- Model: `bert-base-uncased` (Hugging Face)
- Fine-tuned with `BCEWithLogitsLoss`
- Activation: `sigmoid`
- Optimizer: AdamW + Scheduler
- Metrics: F1-score (macro/micro), precision, recall

---

## 📊 Evaluation Metrics

Because multiple labels can be true at once, we use:

- `Micro F1-score`: emphasizes common labels
- `Macro F1-score`: gives equal weight to all labels
- `Accuracy`: exact match (very strict)

---

## 💻 Run Locally

```bash
git clone https://github.com/Koushik7893/text-multilabel-classification-dl-bert
cd text-multilabel-classification
pip install -r requirements.txt
````

---

## 🔮 Future Additions

* ✅ Add attention-based BiLSTM model
* ✅ Integrate label correlation modeling
* 🔄 Explore MultiLabel Transformers like `SetFit`, `XLNet`
* 🔄 Add ONNX export + Hugging Face Model Card

---

## ✍️ Author

**Koushik Reddy**
🔗 [Hugging Face](https://huggingface.co/Koushim) 

---

## 📌 License

This project is open source and available under the [Apache License](LICENSE).

````
