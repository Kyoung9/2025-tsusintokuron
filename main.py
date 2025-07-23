import argparse
from dataloader import get_nslkdd_dfs, get_nslkdd_datasets, get_nslkdd_dataloaders, get_input_dim
from trainer import train_model
from models import MLP
import numpy as np

from preprocess import plot_label_histogram

def main():
    parser = argparse.ArgumentParser(description="Train MLP on NSL-KDD dataset")

    # データ指定
    parser.add_argument("--data_dir", type=str, default="dataset/NSL-KDD")
    parser.add_argument("--train_file", type=str, default="KDDTrain+.txt")
    parser.add_argument("--test_file", type=str, default="KDDTest+.txt")

    # 正規化
    parser.add_argument("--normalize", type=str, choices=["zscore", "minmax", "none"], default="minmax")

    # ラベルモードとOne-hotエンコード
    parser.add_argument("--label_mode", type=str, choices=["binary", "4class", "full"], default="4class")
    parser.add_argument("--one_hot", dest="one_hot", action="store_true", help="Enable one-hot encoding for categorical features")
    parser.add_argument("--no-one_hot", dest="one_hot", action="store_false", help="Disable one-hot encoding")
    parser.set_defaults(one_hot=True)  # デフォルトはTrue

    #
    parser.add_argument('--log_dir', type=str, default='logs', help='Base directory to save logs')
    parser.add_argument('--experiment_name', type=str, default='exp1', help='Name of this experiment')

    # 学習設定
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument('--confusion_matrix_path', type=str, default='logs/confusion_matrix.png')
    parser.add_argument('--report_csv_path', type=str, default='logs/classification_report.csv')
    parser.add_argument('--tensorboard_logdir', type=str, default='runs/exp1')


    args = parser.parse_args()

    normalize = args.normalize != "none"

    print(f"Creating dataframe...")
    train_df, test_df = get_nslkdd_dfs(
        data_dir=args.data_dir,
        train_file=args.train_file,
        test_file=args.test_file,
        normalize=normalize,
        normalization=args.normalize,
        label_mode = args.label_mode,
        one_hot=args.one_hot,
    )

    # assert set(np.unique(train_df['label'])) == set(np.unique(test_df['label'])), "Train/Test ラベル不一致"

    plot_label_histogram(train_df, label_column="label", title="Train Set Label Histogram")
    plot_label_histogram(test_df, label_column="label", title="Test Set Label Histogram")

    print(f"Setting datasets...")
    train_dataset, test_dataset = get_nslkdd_datasets(
        train_df = train_df,
        test_df = test_df
    )

    print(f"Setting dataloaders...")
    train_loader, test_loader = get_nslkdd_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size
    )

    input_dim = get_input_dim(train_loader)
    from label_utils import get_num_classes
    output_dim = get_num_classes(args.label_mode)
    # model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=[128, 64])
    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=[128, 64], dropout=0.9)

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        label_mode=args.label_mode,
        val_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        experiment_name = args.experiment_name,
        log_dir = args.log_dir

    )

if __name__ == "__main__":
    main()
