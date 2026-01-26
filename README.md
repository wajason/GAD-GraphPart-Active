# GraphPart for Graph Anomaly Detection (GAD)

This repository is a modified implementation of **GraphPart** adapted for **Graph Anomaly Detection (Active Learning)** tasks.
Original paper: [Partition-Based Active Learning for Graph Neural Networks (TMLR 2023)](https://arxiv.org/abs/2201.09391).

## Modifications
- Adapted for GAD datasets (Enron, Weibo, Disney, etc.).
- Switched evaluation metric from Accuracy to **ROC-AUC**.
- Optimized for Graph Anomaly Detection scenarios (Class Imbalance).

## Original Citation
If you use the core GraphPart algorithm, please cite the original authors:
@article{ma2022partition,
  title={Partition-based active learning for graph neural networks},
  author={Ma, Jiaqi and Ma, Ziqiao and Chai, Joyce and Mei, Qiaozhu},
  journal={arXiv preprint arXiv:2201.09391},
  year={2022}
}