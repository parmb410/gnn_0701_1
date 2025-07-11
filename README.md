
# EMG-Based Activity Recognition with GNNs, Curriculum Learning, SHAP & Auto-K

## 📌 Overview

This project extends the paper OUT-OF-DISTRIBUTION REPRESENTATION LEARNING FOR TIME SERIES CLASSIFICATION at ICLR 2023.

Link- https://paperswithcode.com/paper/generalized-representations-learning-for-time . 

**GNNIntegrated** is a modular deep learning framework for **Human Activity Recognition (HAR)** from **Electromyography (EMG)** data. It leverages **Graph Neural Networks (GNNs)**, along with innovative extensions such as:

- **Curriculum Learning**: Progressive sample learning for better generalization.
- **Automated K Estimation**: Dynamic determination of cluster numbers.
- **SHAP Explainability**: Model interpretation via feature attribution.
- **Graph-Based Learning**: Temporal and structural dependencies through GNNs.

Abstract: We address the challenge of Out-of-Distribution (OOD) generalization in time series by enhancing the DIVERSIFY framework with four key components: Curriculum Learning, Graph Neural Networks, SHAP-based explainability, and Automated K Estimation. These additions improve adaptability, interpretability, and scalability. Our method demonstrates mix performance on real-world datasets like EMG and UCI-HAR, achieving better OOD accuracy and generalization. The integrated framework adapts to unseen domains and offers meaningful insights into model decisions, boosting both usability and reliability.

![image](https://github.com/user-attachments/assets/a5da6a74-70e4-4232-9e5b-153e616d297b)

---

## 🧠 Key Features

- **Temporal GCNs** for sequence-aware modeling of EMG signals.
- **Curriculum Learning** pipeline for progressively harder samples.
- **Auto-K Clustering** using Silhouette, CH, DB scores to auto-determine `k`.
- **SHAP Integration** to explain model decisions post-training.
- **Domain Adaptation & Diversification** for robust cross-subject learning.

---

## 🔧 Core Pipelines

### 🏋️ Training Pipeline (`train.py`) 

    
    ┌───────────────────────────────┐
    │        train.py               │
    ├───────────────────────────────┤
    │                               │
    │ 1. Argument Parsing           │
    │    • dataset, model, flags    │
    │                               │
    │ 2. Dataset Preparation        │
    │    • get_act_dataloader()     │
    │                               │
    │ 3. Graph Construction         │
    │    • graph_builder.py         │
    │    • temporal edges           │
    │                               │
    │ 4. Curriculum Learning        │
    │    • reorder samples          │
    │                               │
    │ 5. Model Initialization       │
    │    • ActNetwork (GNN)         │
    │    • domain adaptation opt.   │
    │                               │
    │ 6. Optimization               │
    │    • multi-component loss     │
    │      - alignment              │
    │      - classification         │
    │      - diversification        │
    │                               │
    │ 7. Logging                    │
    │    • loss, accuracy           │
    │    • clustering metrics       │
    │      - Silhouette             │
    │      - Calinski-Harabasz (CH) │
    │      - Davies-Bouldin (DB)    │
    └───────────────────────────────┘


### 📊 Evaluation Pipeline
    
    ┌───────────────────────────────┐
    │    evaluation pipeline        │
    ├───────────────────────────────┤
    │                               │
    │ 1. Load Best Model            │
    │    • resume checkpoint        │
    │                               │
    │ 2. Feature Extraction         │
    │    • GNN encoder embeddings   │
    │                               │
    │ 3. Auto-k Clustering          │
    │    • optimal cluster number   │
    │    • Silhouette, CH, DB       │
    │                               │
    │ 4. Classification             │
    │    • logistic regression      │
    │    • or NN classifier         │
    │                               │
    │ 5. SHAP Analysis              │
    │    • shap_utils.py            │
    │    • feature attributions     │
    │                               │
    │ 6. Visualization              │
    │    • confusion matrix         │
    │    • silhouette plots         │
    │    • SHAP explanations        │
    └───────────────────────────────┘


---

## 📁 File Structure

```
gnnintegrated-main/
├── train.py                  # Main pipeline
├── shap_utils.py            # SHAP explainability
├── env.yml                  # Environment setup
├── alg/                     # Domain adaptation, optimization
├── datautil/                # EMG dataset handling, clustering
├── gnn/                     # Graph construction, GNN models
├── loss/                    # Custom loss functions
├── network/                 # ActNetwork, adversarial nets
├── utils/                   # Arguments, reproducibility
└── README.md
```

---

## 📦 Dataset: EMG

The project is designed around **EMG (Electromyography) data**, processed into temporal sequences and graphs for HAR. Dataset loading is managed by:

```
datautil/actdata/
  ├── cross_people.py
  ├── util.py
```

Ensure your EMG data is structured or converted accordingly.

Direct link - https://wjdcloud.blob.core.windows.net/dataset/diversity_emg.zip
---

## 🧪 Extensions Breakdown

### 1. **GNN Backbone**

- Implements **Temporal GCNs** in `gnn/temporal_gcn.py`.
- Graphs constructed using `graph_builder.py`.

### 2. **Curriculum Learning**

- Sample difficulty progression implemented in `get_curriculum_loader()`.

### 3. **Automated K Estimation**

- Uses clustering metrics like:
  - **Silhouette Score**
  - **Calinski-Harabasz Index**
  - **Davies-Bouldin Score**
- Implemented in `train.py` and `datautil/cluster.py`.

### 4. **SHAP Integration**

- Post-hoc interpretability via `shap_utils.py`.
- Applies SHAP on feature embeddings for transparency.

---

Optional flags (see `utils/util.py::get_args()`):
- `--curriculum`
- `--enable_shap`
- `--gnn_hidden_dim`
- `--domain_adapt`

---

## 🧾 Outputs and Artifacts

- ✅ Trained model weights
- 📉 Loss and accuracy logs
- 📊 Clustering plots
- 📈 Confusion matrices
- 🔍 SHAP explanations
- 📁 Embedding files for downstream tasks

---

## 📈 Analysis & Visualization

- **Clustering Evaluation**:
  - Automatic `k` tuning
  - CH, DB, and Silhouette plotted
- **Classification Accuracy**:
  - Per-class metrics and confusion matrices
- **SHAP Analysis**:
  - Local/global feature importance
  - Interpretable visual outputs

---

## Limitations and Future Prospects

GNNs proved ineffective for cross-subject EMG classification due to EMG's lack of inherent graph structure, leading to catastrophic training failure and ~16% accuracy. Exploding losses and poor domain alignment indicated architectural mismatch. Latent clustering was unstable, failing domain generalization. In contrast, CNNs like ResNet or TCN exploit EMG’s temporal patterns effectively. GNNs fundamentally require data with inherent graph topology; for tabular or sequential data like EMG, a meaningful graph must first be manually and carefully constructed. Without this, GNNs cannot learn effectively. We recommend removing --use_gnn and using EMG-specific preprocessing with CNNs and domain adaptation methods (e.g., Deep CORAL). GNN integration would require extensive tuning over 3+ months, making CNNs the practical(for short period projects), accurate (80–85%) solution.

---

