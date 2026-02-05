import json
import codecs
import os
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CoraFull, Coauthor
from torch_geometric.nn import GCNConv
from ogb.nodeproppred import PygNodePropPredDataset
import networkx as nx

# [GAD] 引入 PyGOD 載入工具
try:
    from pygod.utils import load_data as load_god_data
except ImportError:
    print("Warning: PyGOD not installed. GAD datasets might fail to load.")

# 引入 GraphPartition 工具
try:
    from partition import GraphPartition
except ImportError:
    print("Warning: partition.py not found. On-the-fly partitioning will fail.")

def load_data(name="cora", read=True, save=True,
              transform=T.ToSparseTensor(),
              pre_compute=True, verbose=False):

    # ==========================================
    # 0. 資料集設定與參數 (GAD Config)
    # ==========================================
    GAD_CONFIG = {
        'enron':    {'k': 12},
        'disney':   {'k': 7},
        'weibo':    {'k': 7},
        'reddit':   {'k': 11},
        'books':    {'k': 12},
        'inj_cora': {'k': 18}
    }
    
    gad_datasets = list(GAD_CONFIG.keys())
    allowed_datasets = ["cora", "pubmed", "citeseer", "corafull",
                        "cs", "physics", 'arxiv'] + gad_datasets

    assert name in allowed_datasets

    path = os.path.join("data", name)
    
    # ==========================================
    # 1. 資料載入 (Data Loading)
    # ==========================================
    if name in gad_datasets:
        if verbose: print(f"Loading GAD dataset: {name}...")
        try:
            data = load_god_data(name)
        except Exception as e:
            if name == 'inj_cora':
                # Fallback for inj_cora if PyGOD name differs
                print("Warning: Failed to load inj_cora directly. Check PyGOD version.")
                raise e
            else:
                raise e

        # [GAD Fix] 格式統一化
        if data.y.dim() > 1:
            data.y = data.y.squeeze()
        data.y = data.y.long()
        data.num_classes = 2
        
        if not hasattr(data, 'adj_t'):
            data = T.ToSparseTensor()(data)
            
    elif name in ['cora', 'pubmed', 'citeseer']:
        dataset = Planetoid(root=path, name=name, transform=transform)
        data = dataset[0]
    elif name == "corafull":
        dataset = CoraFull(root=path, transform=transform)
        data = dataset[0]
    elif name == "cs" or name == "physics":
        dataset = Coauthor(root=path, name=name, transform=transform)
        data = dataset[0]
    elif name == 'arxiv':
        dataset = PygNodePropPredDataset(root=path, name='ogbn-' + name, transform=transform)
        data = dataset[0]
    else:
        raise NotImplementedError(f"Dataset {name} not supported.")

    if not hasattr(data, 'num_classes'):
        data.num_classes = int(data.y.max().item()) + 1
    
    if hasattr(data, 'adj_t') and not isinstance(data.adj_t, torch.Tensor):
        data.adj_t = data.adj_t.to_symmetric()

    # ==========================================
    # [Fix] 2. 設定 Baseline 參數 (避免 AGE 報錯)
    # ==========================================
    # 設定預設參數 (Default Params)
    # age: [alpha, beta, gamma]
    if not hasattr(data, 'params'):
        data.params = {'age': [0.1, 0.1, 0.8]} # 通用預設值

    # 針對特定資料集的微調 (保留原始邏輯)
    if name == 'cora':
        data.max_part = 7
        data.params = {'age': [0.05, 0.05, 0.9]}
    elif name == 'pubmed':
        data.max_part = 8
        data.params = {'age': [0.15, 0.15, 0.7]}
    elif name == 'citeseer':
        data.max_part = 14
        data.params = {'age': [0.35, 0.35, 0.3]}
    elif name in GAD_CONFIG:
        data.max_part = GAD_CONFIG[name]['k']
    else:
        data.max_part = data.num_classes
    

    # ==========================================
    # 2. 設定 Partition 參數
    # ==========================================
    if name in GAD_CONFIG:
        data.max_part = GAD_CONFIG[name]['k']
    elif name == 'cora': data.max_part = 7
    elif name == 'pubmed': data.max_part = 8
    elif name == 'citeseer': data.max_part = 14
    else: data.max_part = data.num_classes

    # ==========================================
    # 3. 建構 NetworkX 圖
    # ==========================================
    if not hasattr(data, 'g'):
        if hasattr(data, 'adj_t'):
            row, col, _ = data.adj_t.coo()
            edges = [(int(i), int(j)) for i, j in zip(row, col)]
        else:
            edges = [(int(i), int(j)) for i, j in zip(data.edge_index[0], data.edge_index[1])]
            
        data.g = nx.Graph()
        data.g.add_edges_from(edges)
        if data.g.number_of_nodes() < data.num_nodes:
            data.g.add_nodes_from(range(data.num_nodes))

    # ==========================================
    # 4. Partition 處理邏輯
    # ==========================================
    partition_file = f"data/partitions/{name}.json"
    loaded_successfully = False

    if read and os.path.exists(partition_file):
        try:
            if verbose: print(f"Loading partitions from {partition_file}")
            with open(partition_file, 'r') as f:
                part_dict = json.load(f)
            
            data.partitions = {}
            for k_str, val in part_dict.items():
                data.partitions[int(k_str)] = torch.tensor(val)
            
            if data.max_part in data.partitions:
                loaded_successfully = True
            else:
                print(f"Warning: Partition file exists but missing K={data.max_part}. Recalculating...")
                
        except Exception as e:
            print(f"Error loading partition file: {e}. Recalculating...")

    if not loaded_successfully:
        print(f"Generating partitions for {name} (K={data.max_part})... This may take a while.")
        
        graph = data.g.to_undirected()
        
        # [Robust Fix] 自動偵測 GraphPartition 接受什麼參數
        # 嘗試 1: 完整參數 (含 log)
        try:
            graph_part = GraphPartition(graph, data.x, data.max_part, name, log=verbose)
        except TypeError:
            # 嘗試 2: 不含 log，但含 name
            try:
                graph_part = GraphPartition(graph, data.x, data.max_part, name)
            except TypeError:
                # 嘗試 3: 最簡參數 (只含 graph, x, k)
                print("Warning: partition.py signature mismatch, trying minimal args...")
                graph_part = GraphPartition(graph, data.x, data.max_part)

        # 執行社群演算法
        communities = graph_part.clauset_newman_moore(weight=None)
        
        # 進行合併
        data.partitions = graph_part.agglomerative_clustering(communities)
        
        # 確保有我們需要的 K
        if data.max_part not in data.partitions:
            available_ks = list(data.partitions.keys())
            if len(available_ks) > 0:
                best_k = min(available_ks, key=lambda x: abs(x - data.max_part))
                print(f"Note: Desired K={data.max_part}, but algo returned K={best_k}. Using K={best_k}.")
                data.max_part = best_k

    # ==========================================
    # 5. 自動存檔
    # ==========================================
    if save and data.partitions is not None:
        if not os.path.exists("data/partitions"):
            os.makedirs("data/partitions")
            
        with open(partition_file, "w", encoding='utf-8') as f:
            part_save = {}
            for key, val in data.partitions.items():
                if isinstance(val, torch.Tensor):
                    part_save[int(key)] = val.cpu().numpy().tolist()
                elif isinstance(val, np.ndarray):
                    part_save[int(key)] = val.tolist()
                else:
                    part_save[int(key)] = val
            json.dump(part_save, f, separators=(',', ':'), sort_keys=True)
            if verbose: print(f"Partitions saved to {partition_file}")

    # ==========================================
    # 6. 特徵平滑預計算
    # ==========================================
    edges = [(int(i), int(i)) for i in range(data.num_nodes)]
    data.g.add_edges_from(edges)

    try:
        if pre_compute:
            feat_dim = data.x.size(1)
            conv = GCNConv(feat_dim, feat_dim, cached=True, bias=False)
            conv.lin.weight = torch.nn.Parameter(torch.eye(feat_dim))
            graph_input = data.adj_t if hasattr(data, 'adj_t') else data.edge_index
            
            with torch.no_grad():
                data.aggregated = conv(data.x, graph_input)
                data.aggregated = conv(data.aggregated, graph_input)
    except Exception as e:
        print(f"Warning: Aggregation failed ({e}), using raw features.")
        data.aggregated = data.x

    # [Fix] 強制修正 GAD 資料集的標籤問題
    # 如果是 GAD 資料集，但標籤卻超過 2 類 (例如 inj_cora 讀成了原始 cora 的 7 類)
    # 我們強制把它轉成二元分類 (One-vs-Rest)
    if name in gad_datasets:
        # 如果發現有超過 1 的標籤 (代表是多類別)
        if data.y.max() > 1:
            if verbose: print(f"Fixing multiclass labels for {name} (Max label: {data.y.max()}) -> Binary")
            
            # 策略：將 0 視為正常，所有非 0 (1~6) 視為異常 (1)
            # 或者：視資料集定義而定，但在異常偵測中通常是把少數類別當 1
            # 這裡我們採用通用的轉換： Class 0 = Normal(0), Others = Anomaly(1)
            data.y = (data.y != 0).long()
            
        # 強制設定類別數為 2，讓模型輸出層正確初始化
        data.num_classes = 2
        
        # 再次確保 y 是 LongTensor
        data.y = data.y.long()

    # [Fix] 確保模型初始化正確
    # 有時候 PyG 讀進來 num_classes 會殘留舊的 (例如 7)
    if name in gad_datasets:
        data.num_classes = 2

    return data

if __name__ == "__main__":
    # 測試腳本：預先生成所有 GAD 資料集的分區檔
    # 如果某個失敗，會印出錯誤但繼續下一個
    for name in ["weibo", "reddit", "books", "inj_cora", "enron", "disney"]:
        try:
            print(f"--- Processing {name} ---")
            data = load_data(name=name, read=False, save=True, verbose=True)
            print(f"[Success] {name} processed. K={data.max_part}\n")
        except Exception as e:
            print(f"[Failed] {name}: {e}\n")