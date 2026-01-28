import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pygod.utils import load_data
from torch_geometric.data import Data

# 1. 定義目標資料集 (依照 GAD-NR 論文)
GAD_DATASETS = ['enron', 'disney', 'weibo', 'reddit', 'books', 'inj_cora']

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def gad_nr_normalization(x):
    """
    嚴格復刻 GAD-NR 的 Min-Max Normalization
    原始代碼:
    node_features_min = node_features.min()
    node_features_max = node_features.max()
    node_features = (node_features - node_features_min)/node_features_max
    """
    x_min = x.min()
    x_max = x.max()
    # 防止分母為 0
    if x_max == 0:
        return x
    return (x - x_min) / x_max

def gad_nr_add_self_loops(edge_index, num_nodes):
    """
    嚴格復刻 GAD-NR 的自環添加邏輯
    原始代碼:
    self_edges = torch.tensor([[i for i in range(num_nodes)],[i for i in range(num_nodes)]])
    edge_index = torch.cat([edge_index,self_edges],dim=1)
    """
    # 產生自環 [0,0], [1,1], ...
    loop_nodes = torch.arange(0, num_nodes, dtype=torch.long)
    self_edges = torch.stack([loop_nodes, loop_nodes], dim=0)
    
    # 強制串接 (即使原本有自環也會再加一次，保持跟 GAD-NR 行為一致)
    # 注意: edge_index 需轉為 CPU 處理以免 device 不一致
    edge_index = edge_index.cpu()
    
    return torch.cat([edge_index, self_edges], dim=1)

def preprocess_and_verify():
    print(f"{'='*20} Phase 1: GAD-NR Style Data Standardization {'='*20}")
    
    stats = []
    
    for name in GAD_DATASETS:
        print(f"\n[Processing] {name}...")
        
        try:
            # A. 下載/載入資料 (PyGOD)
            # PyGOD 的 load_data 已經封裝了 Enron, Disney 等資料集的下載邏輯
            # 對於 inj_cora，PyGOD 會自動處理注入
            data = load_data(name)
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
            print("   (Check network connection or PyGOD version)")
            continue

        # B. 特徵正規化 (Feature Normalization)
        print("   -> Normalizing features (MinMax per GAD-NR)...")
        data.x = gad_nr_normalization(data.x)
        
        # C. 標籤處理 (Label Handling)
        # 邏輯: 只要 data.y != 0 就是異常 (包含 contextual=1, structural=2, both=3)
        # 這是 ROC-AUC 二元分類的標準
        if name == 'inj_cora':
            # 針對 inj_cora 的 bit-encoding 驗證
            # 雖然 .bool() 已經足夠，但我們明確寫出來以示嚴謹
            binary_y = data.y.bool().long()
            print("   -> Handled inj_cora bit-encoded labels.")
        else:
            binary_y = data.y.bool().long()

        # D. 加入自環 (Add Self-loops)
        print("   -> Adding self-loops (GAD-NR style)...")
        num_nodes = data.x.shape[0]
        data.edge_index = gad_nr_add_self_loops(data.edge_index.long(), num_nodes)

        # E. 封裝 (Packaging for GraphPart)
        # GraphPart 的 dataset.py 讀取 .pt 時，期望的是一個 Data 物件
        clean_data = Data(
            x=data.x.float(),
            edge_index=data.edge_index.long(),
            y=binary_y
        )
        
        # F. 存檔
        # GraphPart 預設路徑結構: data/{name}/processed/data.pt
        save_dir = os.path.join('data', name, 'processed')
        ensure_dir(save_dir)
        save_path = os.path.join(save_dir, 'data.pt')
        
        torch.save(clean_data, save_path)
        print(f"   -> Saved to: {save_path}")
        
        # G. 統計驗證
        num_anomalies = clean_data.y.sum().item()
        anomaly_ratio = (num_anomalies / num_nodes) * 100
        
        stats.append({
            'Dataset': name,
            'Nodes': num_nodes,
            'Edges': clean_data.edge_index.shape[1],
            'Anomalies': num_anomalies,
            'Ratio (%)': anomaly_ratio,
            'Features': clean_data.x.shape[1]
        })
        
        print(f"   -> Nodes: {num_nodes}, Anomaly Ratio: {anomaly_ratio:.2f}%")

    return stats

def visualize_stats(stats):
    # 畫個簡單的圖確認資料沒壞掉
    print(f"\n{'='*20} Generating Verification Plot {'='*20}")
    names = [s['Dataset'] for s in stats]
    ratios = [s['Ratio (%)'] for s in stats]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, ratios, color='firebrick', alpha=0.7)
    plt.xlabel('Dataset', fontweight='bold')
    plt.ylabel('Anomaly Ratio (%)', fontweight='bold')
    plt.title('GAD Datasets Anomaly Distribution', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.savefig('gad_data_stats.png')
    print("   -> Saved verification plot to 'gad_data_stats.png'")

if __name__ == "__main__":
    stats_data = preprocess_and_verify()
    if stats_data:
        visualize_stats(stats_data)
        print("\n✅ Phase 1 Complete! Data is strictly aligned with GAD-NR logic.")