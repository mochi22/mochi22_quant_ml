from sklearn.model_selection import KFold
import numpy as np

class PurgedKFold:
    def __init__(self, n_splits=5, purge=0, embargo=0):
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0

        for fold_size in fold_sizes:
            test_start, test_end = current, current + fold_size
            test_indices = indices[test_start:test_end]

            # Purge前後を除外した学習データインデックスを作成
            train_start = 0
            train_end = n_samples
            purge_start = max(train_start, test_start - self.purge)
            purge_end = min(train_end, test_end + self.embargo)

            train_indices = np.concatenate([
                indices[train_start:purge_start],
                indices[purge_end:train_end]
            ])
            yield train_indices, test_indices
            current = test_end

# # 例: PurgedKFold クラスの定義は資料(040.png)に記載されています。
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import KFold

# class PurgedKFold(KFold):
#     def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
#         if not isinstance(t1, pd.Series):
#             raise ValueError('t1 must be a pd.Series')
#         super().__init__(n_splits, shuffle=False, random_state=None) # 時系列なのでshuffleはFalse
#         self.t1 = t1
#         self.pctEmbargo = pctEmbargo

#     def split(self, X, y=None, groups=None):
#         if (X.index == self.t1.index).all() is False:
#             raise ValueError('X and t1 must have the same index')
#         indices = np.arange(X.shape)
#         mbrg = int(X.shape * self.pctEmbargo)
        
#         # KFold の通常のsplitメソッドを呼び出す
#         for i, (train_indices, test_indices) in enumerate(super().split(X, y, groups)):
#             train_indices = train_indices.copy()
#             test_indices = test_indices.copy()
            
#             # パージング処理 (train_timesとtest_timesの重複を削除)
#             # 資料のgetTrainTimesロジックをクラス内に組み込む
#             test_times = self.t1.iloc[test_indices]
#             train_times_initial = self.t1.iloc[train_indices]
            
#             # パージング: test_times と overlapping する train_times_initial を削除
#             # ここは資料の getTrainTimes(t1, testTimes) のロジックを内部で実行する部分です。
#             # 具体的な実装は資料の getTrainTimes のスニペットを参照してください [8]。
#             # 簡単化された例：
#             # 例えば、test_indicesの範囲にあるt1の観測期間と重複するtrain_indicesを削除
            
#             # エンバーゴ期間の適用
#             # 資料の getEmbargoTimes スニペットのロジックを内部で実行する部分です。
#             # test_indicesの直後のmbrg期間をtrain_indicesから削除
            
#             # スニペット getTrainTimes(t1, testTimes) [8] と getEmbargoTimes(t1, pctEmbargo) [3] のロジックを
#             # PurgedKFold クラスの split メソッド内で適切に呼び出す必要があります。
#             # 資料のコードスニペットの完全な統合が必要ですが、ここでは概念的な説明に留めます。
            
#             # 簡略化されたロジック（実際の実装は資料を参照）
#             # train_indicesからtest_timesと重複する観測値を削除
#             # train_indicesからtest_timesの直後のエンバーゴ期間の観測値を削除
            
#             # 以下は資料のPurgedKFoldクラスのsplitメソッドの抜粋 [3]
#             # 実際のロジックは self.t1 と test_indices, train_indices を用いて実行されます。
#             # test_times = self.t1.iloc[test_indices]
#             # train_indices = train_indices[~self.t1.iloc[train_indices].isin(test_times.index)] # 簡略化された重複削除
            
#             # embg = self.t1.iloc[test_indices].max() + self.pctEmbargo * (self.t1.max() - self.t1.min()) # エンバーゴ終了時刻の簡易計算
#             # train_indices = train_indices[self.t1.iloc[train_indices] < embg] # エンバーゴ期間前の訓練データを保持
            
#             # 資料のPurgedKFoldの実装は、これらのステップを正確に実行します [3]。
            
#             yield train_indices, test_indices










# # class PurgedKFold ( BaseKFold):
# # '''
# # 区間にまたがるラベルに対して機能するようにKFoldクラスを拡張する

# # にある訓練データのエンバーゴを示している。スニペット7.2はエンバーゴの

# # 訓練データのうちテストラベル区問と重複する観測値がパージされる

# # ロジックを実装している


# # テストデータセットは連続的(shuffle=False)で、問に訓練データがな
# # いとする
# # '''

# # def _init_(self,n_splits=3,t1=None,pctEmbargo=0.) :
# #     if not isinstance (t1,pd.Series) :
# #         raise ValueError ('Label Through Dates must be a pd.Series')
# #     super (PurgedKFold,self)._ init_ (n_splits,shuffle=False,random_state=None)
# #     self.t1=t1
# #     self.pctEmbargo=pctEmbargo

# # def split (self,X,y=None,groups=None) :

# #     if (X.index == self.t1.index) .sum () != len (self.t1) :
# #         raise ValueError ('X and ThruDateValues must have the same index')

# #     indices=np.arange (X.shape [0])
# #     mbrg=int (X.shape [0] *self.pctEmbargo)

# #     test_starts= [ (i [o] ,i [-1] +1) for i in np.array_split (np.arange (X.shape [0]) self.n_splits)]
# #     for i j in test_starts:
# #         tO=self.t1.index [i] #テストデータセットの始まり
# #         test_indices=indices [i:j]
# #         maxT1ldx=self.t1.index.searchsorted (self.t1 [test_indices].max ())

# #         train_indices=self.t1.index.searchsorted (self.t1 [self.t1 <= t0] .index)

# #         train_indices=np.concatenate ((train_indices,indices [maxT1ldx+mbrg:]))
# #         yield train_indices,test_indices