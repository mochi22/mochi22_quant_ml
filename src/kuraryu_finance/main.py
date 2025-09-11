from train import LightGBMRegressorTrainer
from cv import PurgedKFold
import pandas as pd


if __name__ == "__main__":

    # データ準備
    df = pd.read_csv("labeled_df.csv")  # あなたの DataFrame

    # 特徴量・目的変数指定
    features = df.columns.drop("label")
    target = 'label'

    # インスタンス作成
    trainer = LightGBMRegressorTrainer(features=features, target=target)
    pkf = PurgedKFold(n_splits=3, purge=5, embargo=5)

    # 学習・評価実行
    trainer.run_cv(df, kfold=pkf)