# GraphPart for Graph Anomaly Detection (Active Learning)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux-orange)](https://www.kernel.org/)

This repository is an extended implementation of **GraphPart**, adapted specifically for **Graph Anomaly Detection (GAD)** tasks using Active Learning.

We integrate the partition-based active learning strategy into transductive GAD scenarios, addressing key challenges such as extreme class imbalance and cold-start labeling. This framework supports comprehensive benchmarking across multiple GAD datasets using standard Graph Neural Networks (GCN, GraphSAGE, GAT).

> **Original Paper:** [Partition-Based Active Learning for Graph Neural Networks (TMLR 2023)](https://arxiv.org/abs/2201.09391)

## 🚀 Key Improvements & Features

Compared to the original implementation (focused on node classification), this repository introduces the following adaptations for **Anomaly Detection**:

* **GAD Benchmarks Integration**: Full support for 6 standard GAD datasets: `Weibo`, `Reddit`, `Books`, `Enron`, `Disney`, and `Inj_Cora`.
* **Imbalance-Aware Evaluation**: Switched evaluation metrics from Accuracy/Macro-F1 to **ROC-AUC** and **Average Precision (AP)** to correctly assess performance on highly imbalanced data.
* **Algorithmic Robustness**:
    * Implemented "Crash Protection" for K-Means clustering when partition sizes are smaller than the query budget.
    * Unified label formatting (Binary: 0 for Normal, 1 for Anomaly).
* **Automated Experiment Pipeline**:
    * **10-Seed Stability**: Automated execution over 10 random seeds with mean $\pm$ std reporting.
    * **Resume Capability**: Smart checkpointing to skip completed experimental configurations.
    * **Visualization Tools**: Includes scripts for generating Latex tables (`latex.py`) and plotting learning curves (`plot.py`).

## 💻 System Requirements

**Note:** This codebase is designed for **Linux systems**.
Due to the discontinuation of Windows support for recent versions of **DGL (Deep Graph Library)**, we strongly recommend running this framework on Ubuntu or WSL2.

## 🛠️ Usage

### 1. Installation
Clone the repository and install the dependencies:

```bash
git clone [https://github.com/wajason/GAD-GraphPart-Active.git](https://github.com/wajason/GAD-GraphPart-Active.git)
cd GAD-GraphPart-Active

# It is recommended to create a conda environment
conda create -n graphpart python=3.9
conda activate graphpart

# Install PyTorch and DGL (Linux) matching your CUDA version
# Example:
pip install torch torchvision torchaudio
pip install dgl -f [https://data.dgl.ai/wheels/cu118/repo.html](https://data.dgl.ai/wheels/cu118/repo.html)
pip install torch-geometric ogb scikit-learn matplotlib networkx
```

### 2. Run Experiments
You can run the full benchmark (all datasets, models, and budgets) using the main script. The script automatically handles data partitioning and result logging.

```bash
python main.py
```

### 3. Visualization & Reporting
After the experiments are finished, you can generate reports using the included tools:

* **Generate LaTeX Tables:** Prints IEEE/ACM standard tables for your paper.
    ```bash
    python latex.py
    ```
* **Plot Learning Curves:** Generates `.png` figures for AUC trends across budgets.
    ```bash
    python plot.py
    ```

## 📊 Experimental Results

We evaluate our framework on benchmark datasets. The following learning curves demonstrate that **GraphPart** (Ours) consistently outperforms baselines, especially in low-budget scenarios.


 <img width="400" alt="result_disney_gat" src="https://github.com/user-attachments/assets/f37f1a38-9533-4f13-b515-d28c0aa4340d" />   <img width="400" alt="result_disney_sage" src="https://github.com/user-attachments/assets/4e49e061-e1de-458a-946f-6082d3c5b311" /> 
 <img width="400" alt="result_reddit_gat" src="https://github.com/user-attachments/assets/0275d4aa-d36e-4c67-bdbb-6e38d6221634" />  <img width="400" alt="result_reddit_sage" src="https://github.com/user-attachments/assets/a13633b5-ec1a-4349-b176-eaa793d95a0f" /> 


## 📝 Citation

If you use the core GraphPart algorithm in your research, please cite the original authors:

```bibtex
@article{ma2022partition,
  title={Partition-based active learning for graph neural networks},
  author={Ma, Jiaqi and Ma, Ziqiao and Chai, Joyce and Mei, Qiaozhu},
  journal={arXiv preprint arXiv:2201.09391},
  year={2022}
}
```

If you find this GAD adaptation useful, please star this repository!! Thanks!
