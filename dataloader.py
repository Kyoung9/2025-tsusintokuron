from preprocess import load_nslkdd_dataframe
from torch.utils.data import DataLoader, TensorDataset
import torch


def get_input_dim(dataloader: DataLoader):
    # 最初のバッチで特徴量次元を推定
    x, _ = next(iter(dataloader))
    input_dim = x.shape[1]
    return input_dim

def get_nslkdd_dfs(data_dir,
                   train_file,
                   test_file,
                   normalize,
                   normalization,
                   one_hot,
                   label_mode):

        train_path = f"{data_dir}/{train_file}"
        test_path = f"{data_dir}/{test_file}"

        train_df, test_df = load_nslkdd_dataframe(
            train_path, test_path,
            normalize=normalize,
            normalization=normalization,
            one_hot=one_hot,
            label_mode=label_mode
        )

        if list(train_df.columns) != list(test_df.columns):
            raise ValueError("❌ train_dfとtest_dfの列の順序が一致していません。")
        else:
            print("✅ train_dfとtest_dfの列の順序は一致しています。")

        # 存在する列のみを安全に drop する
        drop_cols = [col for col in ['label_raw', 'label_category'] if col in train_df.columns]
        train_df = train_df.drop(columns=drop_cols)

        drop_cols = [col for col in ['label_raw', 'label_category'] if col in test_df.columns]
        test_df = test_df.drop(columns=drop_cols)

        return train_df, test_df

def get_nslkdd_datasets(train_df,
                        test_df):

        drop_cols = [col for col in ['label'] if col in train_df.columns]
        X_train = torch.tensor(train_df.drop(columns=drop_cols).values, dtype=torch.float32)
        y_train = torch.tensor(train_df['label'].astype(int).values, dtype=torch.long)

        drop_cols = [col for col in ['label'] if col in test_df.columns]
        X_test = torch.tensor(test_df.drop(columns=drop_cols).values, dtype=torch.float32)
        y_test = torch.tensor(test_df['label'].astype(int).values, dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        return train_dataset, test_dataset

def get_nslkdd_dataloaders(train_dataset,
                           test_dataset,
                           batch_size):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description="NSL-KDD Data Loader")

    parser.add_argument("--data_dir", type=str, default="dataset/NSL-KDD", help="Path to data directory")
    parser.add_argument("--train_file", type=str, default="KDDTrain+.txt", help="Train file name")
    parser.add_argument("--test_file", type=str, default="KDDTest+.txt", help="Test file name")
    parser.add_argument("--one_hot", action="store_true", help="Enable one-hot encoding")
    parser.add_argument("--no-one_hot", dest="one_hot", action="store_false", help="Disable one-hot encoding")
    parser.set_defaults(one_hot=True)

    parser.add_argument("--normalize", type=str, default="none", choices=["none", "zscore", "minmax"], help="Normalization method")
    parser.add_argument("--label_mode", type=str, default="4class", choices=["binary", "4class", "full"], help="Label classification mode")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, args.train_file)
    test_path = os.path.join(args.data_dir, args.test_file)
    normalize = None if args.normalize == "none" else args.normalize

    print(f"Loading data from {train_path} and {test_path}")
    print(f"One-hot encoding: {args.one_hot}")
    print(f"Normalization: {normalize}")
    print(f"Label mode: {args.label_mode}")

    train_loader, test_loader = get_nslkdd_dataloaders(
        data_dir=args.data_dir,
        train_file=args.train_file,
        test_file=args.test_file,
        normalize=normalize,
        normalization=args.normalize,
        label_mode = args.label_mode,
        one_hot=args.one_hot,
        batch_size=args.batch_size
    )

    x, y = next(iter(train_loader))

if __name__ == "__main__":
    main()
