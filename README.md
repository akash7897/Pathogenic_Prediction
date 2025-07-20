# Pathogenicity Prediction using Graph Neural Networks (GNN)

## 🧬 Project Overview

This project aims to predict the **pathogenic potential** of genomic mutations using **Graph Neural Networks (GNNs)**. It leverages the spatial and relational structure of gene sequences to determine whether a mutation is likely to be pathogenic or benign.

The goal is to support early detection of harmful mutations in genomic data, enhancing research and diagnostics in genomic medicine.

## 🧠 Model Architecture

- **Graph Construction**: Gene sequences are represented as graphs, where nodes denote nucleotides or mutation positions, and edges represent their biological proximity or functional interaction.
- **Model Used**: 
  - Graph Convolutional Network (GCN)
  - Input: Graphs from mutated genomic data
  - Output: Binary label – *Pathogenic* or *Benign*
- **Loss Function**: Binary Cross Entropy
- **Optimizer**: Adam

## 🗂️ Dataset Description

- **Source**: SCN2A / KCNQ2 gene data from the ClinVar or similar databases.
- **Format**: Sequence and mutation position data, processed into graph structures.
- **Classes**:
  - `0`: Benign mutation
  - `1`: Pathogenic mutation

Each data sample is represented as a graph saved in a format (e.g., `.pt`, `.txt`, or `.npz`), used for training and testing the GNN.

## 📊 Evaluation Metrics

- Accuracy
- Precision & Recall
- ROC-AUC Curve
- Confusion Matrix

Comparisons are made with baseline models like:
- Support Vector Machine (SVM)
- Convolutional Neural Networks (CNN)

## 🧪 Requirements

Install dependencies using pip:
```bash
pip install -r requirements.txt
