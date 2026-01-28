import torch
import numpy as np
from pygod.utils import load_data
from torch_geometric.utils import degree

# 設定要檢查的資料集
datasets = ['enron', 'disney', 'weibo', 'reddit', 'books', 'inj_cora']

print(f"{'Dataset':<10} | {'Nodes':<8} | {'Edges':<8} | {'Feat':<5} | {'Avg Deg':<8} | {'Anomalies':<9} | {'Ratio':<8} | {'Label dist.'}")
print("-" * 110)

for name in datasets:
    try:
        # 1. 載入原始資料
        data = load_data(name)
        
        # 2. 計算基礎統計
        num_nodes = data.x.shape[0]
        num_edges = data.edge_index.shape[1]
        num_features = data.x.shape[1]
        
        # 計算平均分支度 (Avg Degree)
        # PyG 的 edge_index 通常是雙向的 (Directed representation of undirected graph)
        # 所以 edge_index.shape[1] 已經是 2*E (如果是無向圖)
        # 平均度數 = 總邊數 (Edge Index長度) / 節點數
        d = degree(data.edge_index[0], num_nodes)
        avg_deg = d.mean().item()
        
        # 3. 標籤深度分析
        unique_labels, counts = torch.unique(data.y, return_counts=True)
        label_dist = dict(zip(unique_labels.tolist(), counts.tolist()))
        
        # 計算異常數 (非 0 即異常)
        if name == 'inj_cora':
             # inj_cora 是 bit-encoded (1, 2, 3 都是異常)
            binary_y = data.y.bool().long()
        else:
            binary_y = data.y.bool().long()
            
        num_anomalies = binary_y.sum().item()
        ratio = (num_anomalies / num_nodes) * 100
        
        # 4. 輸出結果
        label_str = str(label_dist)
        print(f"{name:<10} | {num_nodes:<8} | {num_edges:<8} | {num_features:<5} | {avg_deg:<8.2f} | {num_anomalies:<9} | {ratio:<7.2f}% | {label_str}")

    except Exception as e:
        print(f"{name:<10} | Error: {e}")

print("-" * 110)
print("註解：")
print("1. Label dist: 顯示原始標籤的分佈。通常 0=正常。")
print("2. 如果 Weibo 的 Label dist 只有 {0: 8xxx, 1: 347}，代表原始資料就只有這些異常。")
print("3. 如果包含 {0:..., 1:..., 2:..., 3:...}，則我們的轉換邏輯會自動將 1,2,3 加總。")