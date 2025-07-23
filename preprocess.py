# preprocess.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from constants import COLUMN_NAMES, CATEGORICAL_COLUMNS, CATEGORICAL_VALUES, CLASS_MAP_FULL, CLASS_MAP_4_NAME, CLASS_MAP_4, CLASS_MAP_BINARY
from feature_config import INCLUDE_FEATURES, EXCLUDE_FEATURES, FEATURE_SELECTION_MODE, LABEL_COLUMNS

import matplotlib.pyplot as plt
from collections import Counter

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    if FEATURE_SELECTION_MODE == "exclude":
        cols_to_drop = [col for col in EXCLUDE_FEATURES if col in df.columns and col not in LABEL_COLUMNS]
        df = df.drop(columns=cols_to_drop)
    elif FEATURE_SELECTION_MODE == "include":
        keep_cols = list(set(INCLUDE_FEATURES + LABEL_COLUMNS))
        df = df[[col for col in df.columns if col in keep_cols]]
    else:
        raise ValueError(f"Invalid FEATURE_SELECTION_MODE: {FEATURE_SELECTION_MODE}")
    return df

def get_filtered_categorical_columns(df: pd.DataFrame) -> list:
    """
    CATEGORICAL_COLUMNS から除外対象を取り除いたものを返す
    """
    if FEATURE_SELECTION_MODE == "exclude":
        return [col for col in CATEGORICAL_COLUMNS if col not in EXCLUDE_FEATURES and col in df.columns]
    elif FEATURE_SELECTION_MODE == "include":
        return [col for col in CATEGORICAL_COLUMNS if col in INCLUDE_FEATURES and col in df.columns]
    else:
        raise ValueError(f"Invalid FEATURE_SELECTION_MODE: {FEATURE_SELECTION_MODE}")

def save_dataframe_to_csv(df: pd.DataFrame, input_path: str, output_dir: str):
    """元のファイル名に基づいてデータフレームをCSVで保存する"""
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_processed.csv")

    df.to_csv(output_path, index=False)

def normalize_dataframe(df_train: pd.DataFrame,
                        df_test: pd.DataFrame,
                        normalization: str = 'zscore',
                        exclude_cols: list = ['label', 'label_raw']) -> tuple:
    """
    指定されたカラムを除外して正規化を行うユーティリティ関数

    Args:
        df_train (pd.DataFrame): 学習用データフレーム
        df_test (pd.DataFrame): テスト用データフレーム
        normalization (str): 'zscore' または 'minmax'
        exclude_cols (list): 正規化対象外の列名リスト

    Returns:
        df_train_norm, df_test_norm: 正規化後のデータフレーム（元と同じ構造）
    """
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]

    if normalization == 'zscore':
        scaler = StandardScaler()
    elif normalization == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported normalization type: choose 'zscore' or 'minmax'")

    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()

    df_train_scaled[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test_scaled[feature_cols] = scaler.transform(df_test[feature_cols])

    return df_train_scaled, df_test_scaled

def load_nslkdd_dataframe(
        train_path,
        test_path,
        normalize=False,
        normalization='zscore',  # 'zscore' or 'minmax'
        one_hot=True,
        label_mode='4class'
):
    """NSL-KDDデータセットの読み込みと前処理を行う

    Args:
        train_file (str): 学習データファイル名
        test_file (str): テストデータファイル名
        normalize (bool): 数値正規化を行うかどうか
        normalization (str): 正規化手法（'zscore' or 'minmax'）
        one_hot (bool): カテゴリカル変数をOne-hotエンコードするかどうか

    Returns:
        df_train, df_test: 前処理済みのデータフレーム（ラベル列は 'label_raw'）
    """

    df_train = select_features(pd.read_csv(train_path, names=COLUMN_NAMES))
    df_test = select_features(pd.read_csv(test_path, names=COLUMN_NAMES))

    # ラベル列を分離して保存
    df_train['label_raw'] = df_train['label']
    df_test['label_raw'] = df_test['label']

    # ラベル変換処理
    print("Converting labels...")
    if label_mode == '4class':
        df_train['label_category'] = df_train['label_raw'].map(CLASS_MAP_4_NAME)
        df_test['label_category'] = df_test['label_raw'].map(CLASS_MAP_4_NAME)
        df_train['label'] = df_train['label_category'].map(CLASS_MAP_4)
        df_test['label'] = df_test['label_category'].map(CLASS_MAP_4)
        df_train = df_train.drop(columns='label_category')
        df_test = df_test.drop(columns='label_category')
    elif label_mode == 'binary':
        df_train['label'] = df_train['label_raw'].map(CLASS_MAP_BINARY)
        df_test['label'] = df_test['label_raw'].map(CLASS_MAP_BINARY)
    elif label_mode == 'full':
        df_train['label'] = df_train['label_raw'].map(CLASS_MAP_FULL)
        df_test['label'] = df_test['label_raw'].map(CLASS_MAP_FULL)
    else:
        raise ValueError(f"Invalid label_mode: {label_mode}")    

    cat_cols = get_filtered_categorical_columns(df_train)

    if one_hot:
        # One-hotエンコード（全列の取りうる値に基づいて統一的に処理）
        print("Expanding categorical variables into one-hot encoding...")
        for col in cat_cols:
            for val in CATEGORICAL_VALUES[col]:
                df_train[f'{col}_{val}'] = (df_train[col] == val).astype(int)
                df_test[f'{col}_{val}'] = (df_test[col] == val).astype(int)
            df_train = df_train.drop(columns=[col])
            df_test = df_test.drop(columns=[col])
    else:
        # 数値ラベルに変換
        print("Converting categorical variables...")
        for col in cat_cols:
            val_to_index = {v: i for i, v in enumerate(CATEGORICAL_VALUES[col])}
            df_train[col] = df_train[col].map(val_to_index)
            df_test[col] = df_test[col].map(val_to_index)

    # 正規化（オプション）
    print("Normalizing...")
    if normalize:
        df_train, df_test = normalize_dataframe(df_train, df_test, normalization=normalization)

    df_test = df_test[df_train.columns]

    save_dataframe_to_csv(df_train, train_path, "processed_csv")
    save_dataframe_to_csv(df_test, test_path, "processed_csv")

    return df_train, df_test


def plot_label_histogram(df: pd.DataFrame, label_column: str = "label", title: str = "Label Distribution") -> None:

    label_counts = Counter(df[label_column])

    # labels = list(label_counts.keys())
    # counts = list(label_counts.values())
    #
    sorted_items = sorted(label_counts.items(), key=lambda x: x[0])
    labels = [label for label, _ in sorted_items]
    counts = [count for _, count in sorted_items]

    total = sum(counts)


    print(f"labels:{labels}")
    print(f"counts:{counts}")
    # ターミナル出力
    print(f"\n📊 {title}")
    for label, count in zip(labels, counts):
        percent = count / total * 100
        print(f"  {label:<15}: {count:6d} ({percent:5.2f}%)")

    # プロット
    plt.figure(figsize=(6 + 0.3 * len(labels), 4))
    bars = plt.bar(labels, counts, color='skyblue')
    plt.title(title)
    plt.xlabel("Label")
    plt.ylabel("Count")

    # 各バーの上に件数と割合を表示
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,
                 height + total * 0.01,
                 f"{count}\n({count / total * 100:.1f}%)",
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()
