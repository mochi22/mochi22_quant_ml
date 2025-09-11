# 下記で任意のグラフを頑張ろうとしたけどめんどいので途中でやめた。

def minus(a, b, round_num):
    return round(a-b, round_num)

def adjust_row_domain(row_heights, vertical_spacing, target_row, shift):
    """
    特定の行だけ上下に shift するための yaxis domain を計算する

    Parameters
    ----------
    row_heights : list of float
        各行の高さ比率 (合計1.0)
    target_row : int
        0始まりの対象行インデックス
    shift : float
        上方向に正, 下方向に負の移動量 (0~1の範囲)

    Returns
    -------
    domains : list of dict
        各 yaxis の domain dict のリスト
    """
    n_rows = len(row_heights)
    total_spacing = vertical_spacing * (n_rows - 1)  # 行間合計
    total_height = sum(row_heights) + total_spacing
    normalized_heights = [round(h / total_height, 2) for h in row_heights]  # 合計1になるよう正規化
    normalized_spacing = round(vertical_spacing / total_height, 2)
    print("normalized_heights:",normalized_heights)
    print("normalized_spacing:",normalized_spacing)

    results = []
    upper = 1.0
    for i, row_height in enumerate(normalized_heights):
        print(i, upper, row_height)
        low = minus(upper, row_height, 2)
        # 対象行に shift を適用
        if i == target_row:
            low = max(0, low+shift)
            upper += shift
        print(i, low, upper)
        results.append([low, upper])
        upper = minus(low, normalized_spacing, 2)  # 次の行の上端
    return results

def set_yaxis_domains(fig, domain_list):
    """
    fig : plotly.graph_objects.Figure
        対象の Figure
    domain_list : list of tuple or list
        各行の domain を [(start1, end1), (start2, end2), ...] の形式で指定
    """
    for i, (start, end) in enumerate(domain_list):
        axis_name = f'yaxis{i+1}'  # yaxis1, yaxis2, ...
        fig.update_layout({axis_name: dict(domain=[start, end])})