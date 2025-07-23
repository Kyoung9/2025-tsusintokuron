# 使いたい特徴量を明示的に指定する場合（include方式）
INCLUDE_FEATURES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes",
    "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    # 必要な項目だけ列挙
]

# 逆に除外したい特徴量を指定する方式もサポート
EXCLUDE_FEATURES = [
    'difficulty'
    # 例: "num_shells", "num_outbound_cmds"
]

# モードを切り替え可能
FEATURE_SELECTION_MODE = "exclude" # "include"  # or "exclude"

# ラベル列は常に保持
LABEL_COLUMNS = ["label", "label_raw", "label_category"]
