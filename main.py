from __future__ import division
from __future__ import print_function

import warnings
# 忽略所有煩人的警告訊息
warnings.filterwarnings("ignore")
import os # 順便補上 os，存檔可能會用到

import argparse
import random
import numpy as np
import torch
from timeit import default_timer as timer

from models import GCN, GAT, SAGE
from query import *
from dataset import load_data


def run(args):

    # [Fix Step 1] 在最外層先讀一次 data，這樣我們才能知道 num_nodes 是多少
    # 這份 data 只用來檢查節點數量，不會用來跑實驗 (實驗會重新讀取乾淨的)
    data = load_data(name=args.dataset, read=True, save=False, verbose=False)

    # [Fix] 確保 args.model 永遠是 List，避免被拆成字母
    if isinstance(args.model, str):
        args.model = [args.model]

    # 讀取目前已完成的進度
    finished_tasks = set()
    if os.path.exists("results.txt"):
        with open("results.txt", "r") as f:
            for line in f:
                if "RESULT:" in line:
                    # 抓取 (dataset|gnn|baseline|B=budget)
                    # 格式範例: RESULT: (weibo|gcn|random|B=10) ...
                    import re
                    match = re.search(r"RESULT: \((.*?)\)", line)
                    if match:
                        finished_tasks.add(match.group(1))
    
    # 外層迴圈：模型 -> 方法 -> 預算
    for gnn in args.model:
        for baseline in args.baselines:
            for budget in args.budget:
                budget = int(budget)

                # [Fix 1] Disney/Enron 防呆：如果預算超過總節點數，就跳過
                # Disney 只有 124 點，跑 160/320 會報錯
                if budget >= data.num_nodes:
                    # 只需要 print 一次提示就好，不用重複 10 次
                    if args.seed == 0: 
                        print(f">>> Skipping Budget {budget} (Exceeds dataset size {data.num_nodes})")
                    continue

                # [Fix] 斷點續傳檢查
                # 組合出當前的任務標籤，例如: weibo|gcn|random|B=10
                current_task_key = f"{args.dataset}|{gnn}|{baseline}|B={budget}"
                
                if current_task_key in finished_tasks:
                    print(f">>> Skipping {current_task_key} (Already Done)")
                    continue  # 直接跳過，跑下一個
                
                # 用來存 10 次實驗的結果
                aucs = []
                aps = []
                founds = [] # 紀錄找到幾個異常點
                
                print(f"\n>>> Running: {gnn} | {baseline} | {args.dataset} | Budget={budget}")

                # [核心修改] 內層迴圈：重複 10 次 (Seeds 0-9)
                for seed in range(10):
                    
                    # [重要] 每次 Loop 都要重新讀取乾淨的 data (避免 Mask 汙染)
                    # verbose=False 讓它安靜一點
                    data = load_data(name=args.dataset,
                                     read=True, save=False, verbose=False).to(args.device)

                    # Set seeds (使用當前的 loop seed)
                    if args.verbose == 1:
                        print('Seed {:03d}:'.format(seed))
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)

                    # Choose model
                    model_args = {
                        "in_channels": data.num_features,
                        "out_channels": data.num_classes,
                        "hidden_channels": args.hidden,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "activation": args.activation,
                        "batchnorm": args.batchnorm
                    }

                    # Initialize models
                    if gnn == "gat":
                        model_args["num_heads"] = args.num_heads
                        model_args["hidden_channels"] = int(args.hidden / args.num_heads)
                        model = GAT(**model_args)
                    elif gnn == "gcn":
                        model = GCN(**model_args)
                    elif gnn == "sage":
                        model = SAGE(**model_args)
                    else:
                        raise NotImplementedError

                    model = model.to(args.device)

                    # General-Purpose Methods
                    if baseline == "random":
                        agent = Random(data, model, seed, args)
                    elif baseline == "density":
                        agent = Density(data, model, seed, args)
                    elif baseline == "uncertainty":
                        agent = Uncertainty(data, model, seed, args)
                    elif baseline == "coreset":
                        agent = CoreSetGreedy(data, model, seed, args)

                    # Graph-specific Methods
                    elif baseline == "degree":
                        agent = Degree(data, model, seed, args)
                    elif baseline == "pagerank":
                        agent = PageRank(data, model, seed, args)
                    elif baseline == "age":
                        agent = AGE(data, model, seed, args)
                    elif baseline == "featprop":
                        agent = ClusterBased(data, model, seed, args,
                                             representation='aggregation',
                                             encoder='gcn')

                    # Our Methods
                    elif baseline == "graphpart":
                        agent = PartitionBased(data, model, seed, args,
                                               representation='aggregation',
                                               encoder='gcn',
                                               compensation=0)
                    elif baseline == "graphpartfar":
                        agent = PartitionBased(data, model, seed, args,
                                               representation='aggregation',
                                               encoder='gcn',
                                               compensation=1)

                    # Ablation Studies
                    elif 'part' in baseline:
                        agent = PartitionBased(data, model, seed, args,
                                               representation=args.representation,
                                               compensation=0)
                    else:
                        agent = ClusterBased(data, model, seed, args,
                                             representation=args.representation)

                    # Initialization
                    training_mask = np.zeros(data.num_nodes, dtype=bool)
                    initial_mask = np.arange(data.num_nodes)
                    np.random.shuffle(initial_mask)
                    init = args.init
                    
                    # 冷啟動策略：這些方法先把 1/3 預算拿去隨機選
                    if baseline in ['density', 'uncertainty', 'coreset', 'age']:
                        init = budget // 3
                    
                    if init > 0:
                        training_mask[initial_mask[:init]] = True

                    training_mask = torch.tensor(training_mask)
                    agent.update(training_mask)
                    agent.train()

                    if args.verbose > 0:
                        # 這裡 evaluate 回傳的是 (auc, ap)
                        auc_init, _ = agent.evaluate()
                        print('Round {:03d}: Labelled: {:d}, Prediction AUC {:.4f}'
                              .format(0, init, auc_init))

                    # Experiment Loop (Active Learning Rounds)
                    # 如果你的 args.rounds 是 1 (一次選完)，這裡只會跑一次
                    current_budget = init
                    
                    for rd in range(1, args.rounds + 1):
                        # Query
                        # 計算這回合該選多少個 (簡單做法：把剩下的一口氣選完，除非 rounds > 1)
                        step_size = (budget - init) // args.rounds
                        if step_size < 1: step_size = budget - current_budget # 防止除法問題

                        if current_budget < budget:
                            start = timer()
                            indices = agent.query(step_size)
                            end = timer()
                            # print('Total Query Runtime [s]:', end - start) # 太吵可以註解掉

                            # Update
                            training_mask[indices] = True
                            agent.update(training_mask)
                            current_budget += len(indices)

                            # Training
                            agent.train()

                    # ==========================================
                    # Evaluate (Seed Loop 結束前的評估)
                    # ==========================================
                    # [修正] 這裡接收 AUC 和 AP (因為我們改過 query.py)
                    auc, ap = agent.evaluate()
                    
                    # 計算找到多少個異常點 (Label=1)
                    train_idx = np.where(agent.data.train_mask.cpu().numpy())[0]
                    y_true = agent.data.y.cpu().numpy()
                    n_found = np.sum(y_true[train_idx] == 1) # 假設 1 是異常

                    aucs.append(auc)
                    aps.append(ap)
                    founds.append(n_found)

                    if args.verbose > 0:
                        print(f"  Seed {seed}: AUC={auc:.4f}, Found={n_found}")

                # ==========================================
                # [統計結果] 10 個 Seeds 跑完後算平均並輸出
                # ==========================================
                auc_mean = np.mean(aucs)
                auc_std = np.std(aucs)
                auc_best = np.max(aucs)
                
                ap_mean = np.mean(aps)
                found_mean = np.mean(founds)
                
                # [你的指定格式]
                # (Dataset|Method|Budget) AUC: mean ± std | Best: max | Found: mean/total
                result_str = (f"RESULT: ({args.dataset}|{gnn}|{baseline}|B={budget}) "
                              f"AUC: {auc_mean:.4f} ± {auc_std:.4f} | "
                              f"Best: {auc_best:.4f} | "
                              f"Found: {found_mean:.1f}/{budget}")
                
                print(result_str)  # 螢幕還是要印
                
                # 自動附加到 results.txt
                with open("results.txt", "a") as f:
                    f.write(result_str + "\n")


if __name__ == '__main__':

    # ==========================================
    # 1. 設定六大 GAD 資料集
    # ==========================================
    GAD_DATASETS = ['weibo', 'reddit', 'books', 'inj_cora', 'enron', 'disney']
    datasets = GAD_DATASETS

    # ==========================================
    # 2. 設定全部 10 個 Baselines
    # ==========================================
    baselines = [
        'random', 
        'uncertainty', 
        'degree', 
        'pagerank', 
        'density', 
        'coreset', 
        'age', 
        'featprop', 
        'graphpart', 
        'graphpartfar'
    ]

    gnns = ['gcn', 'sage', 'gat']
    
    # 預算範圍
    budgets = [10, 20, 40, 80, 160, 320]

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", type=int, default=0, help="Verbose: 0, 1 or 2")
    parser.add_argument(
        "--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # General configs
    parser.add_argument(
        "--baselines", type=list, default=baselines)
    parser.add_argument(
        "--model", default=gnns)
    parser.add_argument(
        "--dataset", default='cora')

    # Active Learning parameters
    parser.add_argument(
        "--budget", type=list, default=budgets,
        help="Number of rounds to run the agent.")
    parser.add_argument(
        "--retrain", type=bool, default=True)
    parser.add_argument(
        "--num_centers", type=int, default=1)
    parser.add_argument(
        "--representation", type=str, default='features')
    parser.add_argument(
        "--compensation", type=float, default=1.0)
    parser.add_argument(
        "--init", type=float, default=0, help="Number of initially labelled nodes.")
    parser.add_argument(
        "--rounds", type=int, default=1, help="Number of rounds to run the agent.")
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of epochs to train.")
    parser.add_argument(
        "--steps", type=int, default=4, help="Number of steps of random walk.")

    # GNN parameters
    parser.add_argument(
        "--seed", type=int, default=0, help="Number of random seeds.")
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4,
        help="Weight decay (L2 loss on parameters).")
    parser.add_argument(
        "--hidden", type=int, default=16, help="Number of hidden units.")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of layers.")
    parser.add_argument(
        "--dropout", type=float, default=0,
        help="Dropout rate (1 - keep probability).")
    parser.add_argument(
        "--batchnorm", type=bool, default=False,
        help="Perform batch normalization")
    parser.add_argument(
        "--activation", default="relu")

    # GAT hyper-parameters
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of heads.")

    args, _ = parser.parse_known_args()

    # ==========================================
    # 執行實驗
    # ==========================================
    for dataset in datasets:
        args.dataset = dataset
        run(args)