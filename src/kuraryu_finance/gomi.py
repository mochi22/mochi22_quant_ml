import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class _BaseKFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        raise NotImplementedError


class PurgedKFold(_BaseKFold):
    def __init__(self, n_splits=3, t1=None, embargo_td=pd.Timedelta(0)):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.embargo = pd.Timedelta(embargo_td)

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(X):
            raise ValueError('X and t1 must have the same index!')

        indices = np.arange(X.shape[0])
        times = X.index

        fold_sizes = np.full(self.n_splits, X.shape[0] // self.n_splits, dtype=int)
        fold_sizes[:X.shape[0] % self.n_splits] += 1
        current = 0

        for fold_size in fold_sizes:
            test_start_idx = current
            test_end_idx = current + fold_size
            test_indices = indices[test_start_idx:test_end_idx]
            test_times = times[test_indices]

            train_indices = np.concatenate([indices[:test_start_idx], indices[test_end_idx:]])

            test_period_start = test_times.min()
            test_period_end = self.t1.loc[test_times].max()

            purged_train_indices = []
            for idx in train_indices:
                obs_start = times[idx]
                if obs_start not in self.t1 or pd.isna(self.t1.loc[obs_start]):
                    continue
                obs_end = self.t1.loc[obs_start]

                if obs_start <= test_period_end and obs_end >= test_period_start:
                    continue  # purge
                purged_train_indices.append(idx)
            train_indices = np.array(purged_train_indices)

            if self.embargo > pd.Timedelta(0):
                embargo_cutoff = test_times.max() + self.embargo
                train_indices = np.array([
                    idx for idx in train_indices if times[idx] > embargo_cutoff
                ])

            train_start_time = times[train_indices].min() if len(train_indices) > 0 else None
            train_end_time = times[train_indices].max() if len(train_indices) > 0 else None

            test_start_time = test_times.min()
            test_end_time = test_times.max()

            purge_start_time = None
            purge_end_time = None
            if len(train_indices) < len(np.concatenate([indices[:test_start_idx], indices[test_end_idx:]])):
                # パージされたデータがあるなら、パージされた期間を計算（簡易的にtest期間とt1で表現）
                purge_start_time = test_period_start
                purge_end_time = test_period_end

            embargo_start_time = test_times.max()
            embargo_end_time = embargo_cutoff

            print(f"Fold {current // fold_size}:")
            print(f"  Train period: {train_start_time} to {train_end_time}")
            print(f"  Test period: {test_start_time} to {test_end_time}")
            if purge_start_time and purge_end_time:
                print(f"  Purge period: {purge_start_time} to {purge_end_time}")
            if embargo_start_time and embargo_end_time:
                print(f"  Embargo period: {embargo_start_time} to {embargo_end_time}")
            print("="*40)

            yield train_indices, test_indices
            current = test_end_idx


def plot_folds(X, t1, cv, embargo_td):
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.get_cmap('tab10', cv.n_splits)
    y_offset = 0

    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        if i==cv.n_splits-1:
            # このとき、トレインは必ず空になるためスキップ
            continue
        train_times = X.index[train_idx]
        test_times = X.index[test_idx]

        ax.plot(train_times, [y_offset] * len(train_times), '|', color=colors(i), label=f'Fold {i+1} - Train')

        ax.plot(test_times, [y_offset + 0.5] * len(test_times), '|', color='red', label='Test' if i == 0 else None)

        embargo_start = test_times.max()
        embargo_end = embargo_start + embargo_td
        ax.axvspan(embargo_start, embargo_end, ymin=(y_offset + 0.25)/cv.n_splits, ymax=(y_offset + 0.75)/cv.n_splits,
                   color='orange', alpha=0.3, label='Embargo' if i == 0 else None)

        y_offset += 1

    ax.set_yticks([])
    ax.set_title("PurgedKFold Splits Visualization")
    ax.set_xlabel("Time")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # データ読み込み
    df = pd.read_csv(
        "labeled_df.csv",
        parse_dates=True,
        index_col=0,
        date_parser=lambda x: pd.to_datetime(x, unit='ms')
    )
    df=df.dropna()
    df.index = pd.to_datetime(df.index, errors='coerce')

    # 一応ソート
    df = df.sort_index()
    print(df)

    embargo_ratio = 1.0 # 高頻度なら小さくしてもよさそう？

    # データの時間幅から自動的に t1 を推定
    time_delta = df.index.to_series().diff().median()
    df['t1'] = df.index + time_delta

    X = df[['open', 'high', 'low', 'close', 'volume']]
    y = df['label']
    t1_series = df['t1']

    # NaNがあるか
    if X.isnull().values.any():
        print("XのNaN数:\n", X.isnull().sum())
    if y.isnull().values.any():
        print("yのNaN数:\n", y.isnull().sum())
    
    # ✅ 直近1か月のデータをテスト期間に固定
    last_date = df.index.max()
    test_start_date = last_date - pd.DateOffset(months=1)

    X_train_cv = X[X.index < test_start_date]
    y_train_cv = y[X.index < test_start_date]

    X_test_cv = X[X.index >= test_start_date]
    y_test_cv = y[X.index >= test_start_date]

    print(f"Train期間: {X_train_cv.index.min()} ～ {X_train_cv.index.max()}")
    print(f"Test期間:  {X_test_cv.index.min()} ～ {X_test_cv.index.max()}")

    embargo_td = time_delta * embargo_ratio
    n_splits = 5
    pkf = PurgedKFold(n_splits=n_splits, t1=t1_series, embargo_td=embargo_td)

    # ✅ 可視化
    plot_folds(X_train_cv, t1_series, pkf, embargo_td)

    # モデル訓練と評価
    model = LogisticRegression(solver='liblinear')
    accuracies = []

    for train_index, test_index in pkf.split(X_train_cv, y_train_cv):
        X_train, X_eval = X_train_cv.iloc[train_index], X_train_cv.iloc[test_index]
        y_train, y_eval = y_train_cv.iloc[train_index], y_train_cv.iloc[test_index]

        if len(train_index) == 0 or len(test_index) == 0:
            print("空のfoldが発生しました。スキップします。")
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_eval)

        accuracy = accuracy_score(y_eval, y_pred)
        accuracies.append(accuracy)

    print(f"\n平均精度: {np.mean(accuracies):.4f}")
    print(f"精度の標準偏差: {np.std(accuracies):.4f}")
