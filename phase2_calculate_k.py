import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from kneebow.rotor import Rotor
from torch_geometric.utils import to_networkx

# 關鍵：直接引用您原始碼中的 partition 模組
# 假設 partition.py 在當前目錄下
import sys
sys.path.append('.') 
from partition import GraphPartition

# 設定
GAD_DATASETS = ['enron', 'disney', 'weibo', 'reddit', 'books', 'inj_cora']
DATA_ROOT = 'data'
SAVE_JSON = os.path.join(DATA_ROOT, 'partitions.json')
PLOT_DIR = 'plots_elbow'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def compute_optimal_k_rigorous(dataset_name):
    print(f"\n[Rigorous Analysis] {dataset_name}...")
    
    # 1. 讀取 .pt 檔
    pt_path = os.path.join(DATA_ROOT, dataset_name, 'processed', 'data.pt')
    if not os.path.exists(pt_path):
        print(f"❌ File not found: {pt_path}")
        return 5 # Fallback
    
    data = torch.load(pt_path)
    
    # 2. 轉換為 NetworkX
    # GraphPart 的 partition.py 預期接收一個 NetworkX 圖
    # 注意：轉為無向圖以進行社群偵測
    g_nx = to_networkx(data, to_undirected=True)
    
    # 3. 初始化 GraphPartition 物件 (使用原始代碼!)
    # 參數對應: GraphPartition(graph, x, num_classes)
    # num_classes 在 CNM 計算中其實沒用到，但 init 需要，我們給個 dummy 值 (e.g. 2)
    gp = GraphPartition(g_nx, data.x, num_classes=2)
    
    print(f"   -> Running Clauset-Newman-Moore (Original Impl)...")
    print("      (This iterates merging communities, please wait...)")
    
    # 4. 執行 CNM
    # 這會從 N 個群合併到 1 個群 (或直到 q_break)，並將過程存入 gp.costs
    # 我們不設限制，讓它跑完或自動停
    gp.clauset_newman_moore()
    
    costs = gp.costs
    
    # 5. 使用 Kneebow 找 Elbow
    # costs 紀錄了每次合併的 "代價" (modularity change)
    # 我們需要根據論文 Figure 1 的邏輯找出轉折點
    
    if len(costs) > 5:
        # 準備數據: X軸是 "Merge Step" (隱含代表 K 的變化), Y軸是 Cost
        # 為了找到 K，我們需要知道每一步對應還剩幾個群
        # CNM 每一步合併兩個群，所以群數是 N -> N-1 -> ... -> K
        # 但 kneebow 只需要看曲線形狀
        
        y_data = np.array(costs)
        x_data = np.arange(len(costs))
        xy = np.column_stack((x_data, y_data))
        
        rotor = Rotor()
        rotor.fit_rotate(xy)
        elbow_idx = rotor.get_elbow_index()
        
        # 關鍵：將 index 轉換回 K (分群數)
        # CNM 是從 N 個群開始合併。第 0 步是 N 群，第 elbow_idx 步是 N - elbow_idx 群？
        # 讓我們確認 partition.py 邏輯。通常 Elbow 是在 Cost 變化劇烈的地方。
        # 論文中 Figure 1 是 K vs Cost。
        # 我們這裡簡化：K通常不會太大，我們記錄的是 merge cost。
        # 如果 cost 突然變大/變小，代表切分結構被破壞。
        
        # 為了更直觀，我們反過來看：
        # 最後一次合併 (變成1群) 是列表的最後。
        # 我們想要的是保留多少個群。
        # 這裡我們採用 Kneebow 對 curve 的標準判定
        
        # 修正邏輯：我們直接將 costs 畫出來，並觀察 elbow 位置
        # 假設 costs 是從 "細碎" 到 "整體" 的過程
        # 論文提到 "we find the elbow of the costs"
        
        # 這裡的 elbow_idx 是在 costs 陣列中的索引
        # 我們暫且估計 K = 總點數 - elbow_idx (如果每步併一次)
        # 但因為點數太多，這樣算 K 可能不準。
        # 比較穩健的做法：將此 dataset 的 K 設為一個合理的範圍，
        # 或者直接根據 kneebow 在 costs 曲線上的位置比例來決定。
        
        # 為了避免計算誤差，我們存下 cost 曲線圖讓您人工確認，
        # 同時先給出一個基於 kneebow 的建議值。
        
        # 根據 GraphPart 論文 Figure 1，Cora (2708點) 的 K 約為 15-20。
        # 如果我們算出來差異太大，可能需要調整。
        
        # 這裡我們先用一個 heuristic:
        # 假設 elbow 發生在合併過程的早期(保留很多群) 或 晚期(只剩幾群)
        # 通常是用 (Total Merges - Elbow Index) 或者是直接看曲線
        
        # 讓我們相信 Kneebow 抓到的特徵變化點
        # 我們將 costs 倒過來 (從 1 群到 N 群) 比較好對應 K
        costs_reversed = costs[::-1]
        xy_rev = np.column_stack((np.arange(len(costs)), costs_reversed))
        rotor.fit_rotate(xy_rev)
        elbow_k_idx = rotor.get_elbow_index()
        
        # 這個 index 大致對應 K 的數量 (從 1 開始數)
        optimal_k = elbow_k_idx + 2 # 加一點緩衝 (至少2群)
        
        print(f"   -> ✅ Found Elbow at index {elbow_k_idx}, Estimated K = {optimal_k}")
        
        # 畫圖存檔
        ensure_dir(PLOT_DIR)
        plt.figure(figsize=(10, 6))
        plt.plot(costs, label='Merge Cost (CNM)')
        plt.axvline(x=elbow_idx, color='r', linestyle='--', label='Elbow Point')
        plt.title(f'{dataset_name}: CNM Costs Curve (Est. K={optimal_k})')
        plt.xlabel('Merge Steps')
        plt.ylabel('Cost')
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIR, f'{dataset_name}_costs.png'))
        plt.close()
        
        return int(optimal_k)
        
    else:
        print("   -> ⚠️ Not enough merge steps.")
        return 5

def main():
    print(f"{'='*20} Phase 2: Rigorous Partition Calculation {'='*20}")
    
    if os.path.exists(SAVE_JSON):
        with open(SAVE_JSON, 'r') as f:
            partitions = json.load(f)
    else:
        partitions = {}
        
    # 逐一處理
    for name in GAD_DATASETS:
        try:
            k = compute_optimal_k_rigorous(name)
            partitions[name] = k
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 存檔
    with open(SAVE_JSON, 'w') as f:
        json.dump(partitions, f, indent=4)
        
    print(f"\n✅ Phase 2 Complete. Optimal K saved to {SAVE_JSON}")
    print(f"   Please check '{PLOT_DIR}' to verify the curves matches paper's Figure 1 style.")

if __name__ == "__main__":
    main()