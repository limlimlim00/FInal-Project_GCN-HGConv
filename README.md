# Comparative Study of GCN and HGConv: Structure, Features, and Noise

This repository contains code and results for the project **"Comparative Study of GCN and HGConv: Structure, Features, and Noise"**, conducted as part of an AI course final project at Seoul National University.

The goal of this project is to systematically compare **Graph Convolutional Network (GCN)** and **Hypergraph Convolution (HGConv)** models under various conditions, focusing on structural generalization and robustness to noisy inputs.

## 📁 Repository Structure

```
.
├── Performance/
│ ├── gcn_vs_hgconv_accuracy_analysis.ipynb
│ └── Figures/ # test accuracy plots (Cora, Citeseer, Pubmed, BAShapes)
├── Perturbation/
│ ├── robustness_generalization_test.ipynb
│ └── Figures/ # test accuracy under noise (Gaussian, edge, label)
├── FinalProject_GCN_vs_HGConv.pdf
└── README.md
```

Each folder contains:

- An `.ipynb` notebook that runs the experiment.
- A `Figures/` directory that stores performance plots under clean and corrupted settings.

## 🧪 Experiments Overview

We evaluated GCN and HGConv models under two experimental settings:

1. **Clean Setting**  
   - Datasets: Cora, Citeseer, Pubmed, BAShapes (with/without features)  
   - Hyperedge types: 1-hop neighbors, k-NN (cosine similarity)  
   - Metric: Test accuracy at best validation performance

2. **Robustness Setting**  
   - Datasets: Cora, Citeseer, Pubmed  
   - Corruptions:
     - **Feature Noise**: Add Gaussian noise (σ=0.1) to node features  
     - **Structural Perturbation**: Randomly add/remove 5% of edges  
     - **Label Noise**: Flip 10% of training labels

All models are 2-layer GCN/HGConv implemented in PyTorch Geometric (PyG).

## 📊 Key Results

| Dataset                | Best Model (Clean)             | Best Model (Feature Noise)     | Best Model (Noisy Overall)     |
|------------------------|--------------------------------|--------------------------------|--------------------------------|
| BAShapes (with feats)  | HGConv-kNN > HGConv-1hop > GCN | HGConv-1hop                    | HGConv-1hop                    |
| Cora                   | GCN ≈ HGConv                   | **HGConv-1hop (+15.5%)**       | HGConv-1hop                    |
| Citeseer               | GCN ≈ HGConv                   | **HGConv-1hop (+8.6%)**        | HGConv-1hop                    |
| Pubmed                 | GCN ≈ HGConv                   | **HGConv-1hop (+15.1%)**       | HGConv-1hop                    |

- **Hyperedge construction plays a critical role** in feature-aware structural tasks.
- **HGConv-1hop shows strong robustness** to input corruption across all datasets.

## 📌 Notes

- All experiments were run on NVIDIA A100 GPU (Colab).
- Hyperparameters: Adam optimizer, 200 epochs, hidden dim=16, dropout=0.5
