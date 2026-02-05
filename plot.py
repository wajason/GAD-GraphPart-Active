import matplotlib.pyplot as plt
import re
import os

def parse_results(filename="results.txt"):
    """
    讀取 results.txt 並解析數據
    結構: data[dataset][model][baseline] = {'x': [budget...], 'y': [auc...]}
    """
    if not os.path.exists(filename):
        print(f"找不到 {filename}，請先執行 main.py 產生結果。")
        return {}

    data = {}
    
    # 解析格式: RESULT: (weibo|gcn|random|B=10) AUC: 0.1136 ...
    pattern = r"RESULT: \((.*?)\|(.*?)\|(.*?)\|B=(\d+)\) AUC: ([\d\.]+) .*"

    with open(filename, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                dataset = match.group(1).strip()
                model = match.group(2).strip()   # 例如 gcn, sage, gat
                baseline = match.group(3).strip()
                budget = int(match.group(4))
                auc = float(match.group(5))

                # 初始化層級結構
                if dataset not in data:
                    data[dataset] = {}
                if model not in data[dataset]:
                    data[dataset][model] = {}
                if baseline not in data[dataset][model]:
                    data[dataset][model][baseline] = {'x': [], 'y': []}
                
                # 存入數據
                data[dataset][model][baseline]['x'].append(budget)
                data[dataset][model][baseline]['y'].append(auc)

    return data

def plot_charts(data):
    """
    生成 18 張圖 (Dataset x Model)
    """
    if not data:
        print("沒有數據可畫圖。")
        return

    # 定義 10 個 Baseline 的顏色與樣式，確保好辨識
    styles = {
        'graphpart':    {'color': '#d62728', 'marker': '*', 'linestyle': '-', 'linewidth': 2.5, 'label': 'GraphPart (Ours)'}, # 紅色星星
        'graphpartfar': {'color': '#ff7f0e', 'marker': 'P', 'linestyle': '-', 'linewidth': 2.5, 'label': 'GraphPart-Far (Ours)'}, # 橘色加號
        
        'random':       {'color': 'gray',    'marker': 'o', 'linestyle': '--', 'linewidth': 1.5},
        'uncertainty':  {'color': '#1f77b4', 'marker': 's', 'linestyle': '-.'}, # 藍色方形
        'density':      {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'}, # 綠色三角形
        'coreset':      {'color': '#9467bd', 'marker': 'v', 'linestyle': '-.'}, # 紫色倒三角
        'degree':       {'color': '#8c564b', 'marker': 'D', 'linestyle': ':'},  # 棕色菱形
        'pagerank':     {'color': '#e377c2', 'marker': 'p', 'linestyle': ':'},  # 粉紅五邊形
        'age':          {'color': '#bcbd22', 'marker': 'h', 'linestyle': ':'},  # 黃綠色六邊形
        'featprop':     {'color': '#17becf', 'marker': 'X', 'linestyle': ':'},  # 青色叉叉
    }

    # 第一層迴圈：Dataset
    for dataset, models in data.items():
        # 第二層迴圈：Model (GCN, SAGE, GAT)
        for model, baselines in models.items():
            
            plt.figure(figsize=(10, 7))
            
            # 設定標題 (例如: Weibo - GCN)
            plt.title(f"Performance on {dataset.upper()} using {model.upper()}", fontsize=16, fontweight='bold')
            plt.xlabel("Budget (Labeled Nodes)", fontsize=14)
            plt.ylabel("ROC-AUC Score", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.5)

            # 畫出 10 條線 (Baselines)
            for method_name, points in baselines.items():
                # 排序 Budget 以確保連線正確
                sorted_pairs = sorted(zip(points['x'], points['y']))
                if not sorted_pairs: continue
                
                x_vals = [p[0] for p in sorted_pairs]
                y_vals = [p[1] for p in sorted_pairs]

                # 取得樣式，如果沒有定義就用預設
                st = styles.get(method_name, {'marker': 'o'})
                label_name = st.get('label', method_name) # 使用自訂標籤名稱

                plt.plot(x_vals, y_vals, 
                         label=label_name, 
                         color=st.get('color'),
                         marker=st.get('marker'),
                         linestyle=st.get('linestyle', '-'),
                         linewidth=st.get('linewidth', 1.5),
                         markersize=8)

            # 圖例放在右下角，避免擋住線
            plt.legend(loc='best', fontsize=10, ncol=2) # 分兩列顯示比較整齊
            
            # 存檔檔名: result_weibo_gcn.png
            output_filename = f"result_{dataset}_{model}.png"
            plt.savefig(output_filename, dpi=300, bbox_inches='tight') # 高解析度存檔
            print(f"[{dataset} - {model}] 圖表已儲存: {output_filename}")
            
            plt.close() # 關閉畫布釋放記憶體

if __name__ == "__main__":
    print("正在讀取 results.txt 並繪製圖表...")
    results = parse_results()
    plot_charts(results)
    print("所有圖表繪製完成！")